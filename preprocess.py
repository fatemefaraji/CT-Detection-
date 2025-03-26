import os
import zipfile
import glob
import pydicom
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from keras.utils import to_categorical
import h5py
import json
import multiprocessing
from multiprocessing import Pool, cpu_count
import gc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MedicalImagePreprocessor:
    def __init__(self, zipPath, metadataPath, outputPath):
        self.zipPath = zipPath
        self.metadataPath = metadataPath
        self.outputPath = outputPath
        self.extractedDir = '/content/extractedData'
        self.annotationsFile = 'annotations.json'

    def extractZipFiles(self):
        os.makedirs(self.extractedDir, exist_ok=True)
        with zipfile.ZipFile(self.zipPath, 'r') as zipRef:
            zipRef.extractall(self.extractedDir)

    def parseXmlFile(self, xmlPath):
        try:
            tree = ET.parse(xmlPath)
            rootXml = tree.getroot()
            namespaces = {'nih': 'http://www.nih.gov'}

            studyUidElement = rootXml.find('.//nih:StudyInstanceUID', namespaces)
            if studyUidElement is None:
                studyUidElement = rootXml.find('.//nih:CXRSeriesInstanceUid', namespaces)

            studyUid = studyUidElement.text if studyUidElement is not None else None

            malignancyScores = [int(nodule.find('.//nih:malignancy', namespaces).text)
                                for nodule in rootXml.findall('.//nih:unblindedReadNodule', namespaces)
                                if nodule.find('.//nih:malignancy', namespaces) is not None]

            return studyUid, max(malignancyScores) if malignancyScores else 0
        except Exception:
            return None, None

    def parseXmlAnnotations(self):
        lidcDir = os.path.join(self.extractedDir, 'LIDC-IDRI')
        xmlFiles = glob.glob(os.path.join(lidcDir, '**/*.xml'), recursive=True)

        with Pool(max(1, cpu_count() - 1)) as pool:
            results = pool.map(self.parseXmlFile, xmlFiles)

        annotations = {studyUid: malignancy for studyUid, malignancy in results if studyUid}

        with open(self.annotationsFile, "w") as f:
            json.dump(annotations, f)

        return annotations

    def loadAnnotations(self):
        if os.path.exists(self.annotationsFile):
            with open(self.annotationsFile, "r") as f:
                return json.load(f)
        return {}

    def processDicomFile(self, dicomPath, studyUid, annotations):
        try:
            ds = pydicom.dcmread(dicomPath)
            image = ds.pixel_array.astype(np.float32)

            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = np.array(tf.image.resize(image[..., np.newaxis], (128, 128)))

            label = 1 if annotations.get(studyUid, 0) > 3 else 0
            return image, label
        except Exception:
            return None, None

    def preprocessImages(self):
        metadataDf = pd.read_csv(self.metadataPath)
        ctSeries = metadataDf[metadataDf['Modality'] == 'CT']

        annotations = self.loadAnnotations()
        if not annotations:
            annotations = self.parseXmlAnnotations()

        slices, labels, studyUids = [], [], []

        for _, row in ctSeries.iterrows():
            studyUid = row['Study UID']
            fileLocation = row['File Location'].lstrip('.\\')
            patientDir = os.path.join(self.extractedDir, fileLocation)

            if os.path.exists(patientDir):
                dicomFiles = glob.glob(os.path.join(patientDir, "*.dcm"))

                with Pool(max(1, cpu_count() // 2)) as pool:
                    results = pool.starmap(self.processDicomFile,
                                           [(dicomFile, studyUid, annotations) for dicomFile in dicomFiles])

                for image, label in results:
                    if image is not None and label is not None:
                        slices.append(image)
                        labels.append(label)
                        studyUids.append(studyUid)

        slices = np.array(slices)
        labels = np.array(labels)

        os.makedirs(os.path.dirname(self.outputPath), exist_ok=True)
        with h5py.File(self.outputPath, 'w') as hdf:
            hdf.create_dataset('slices', data=slices, compression='gzip', chunks=True)
            hdf.create_dataset('labels', data=labels, compression='gzip')
            hdf.create_dataset('studyUids', data=np.array(studyUids, dtype='S'), compression='gzip')

        return slices, labels, studyUids

    def cleanup(self):
        os.system(f'rm -rf {self.extractedDir}')
        os.system(f'rm {self.annotationsFile}')

    def process(self):
        self.extractZipFiles()
        try:
            result = self.preprocessImages()
            return result
        finally:
            self.cleanup()

def preprocessMedicalData(zipPath, metadataPath, outputPath):
    preprocessor = MedicalImagePreprocessor(zipPath, metadataPath, outputPath)
    return preprocessor.process()

def combinePreprocessedFiles(filePaths, finalOutputPath):
    allSlices, allLabels, allStudyUids = [], [], []

    for path in filePaths:
        with h5py.File(path, 'r') as hdf:
            allSlices.append(hdf['slices'][:])
            allLabels.append(hdf['labels'][:])
            allStudyUids.extend(hdf['studyUids'][:])

    finalSlices = np.concatenate(allSlices)
    finalLabels = np.concatenate(allLabels)

    with h5py.File(finalOutputPath, 'w') as hdf:
        hdf.create_dataset('slices', data=finalSlices, compression='gzip', chunks=True)
        hdf.create_dataset('labels', data=finalLabels, compression='gzip')
        hdf.create_dataset('studyUids', data=np.array(allStudyUids, dtype='S'), compression='gzip')

    return finalSlices, finalLabels, allStudyUids

if __name__ == '__main__':
    multiprocessing.freeze_support()

    zipFiles = [
        '/content/data1.zip',
        '/content/data2.zip'
    ]
    metadataPath = '/content/metadata.csv'  # single metadata file
    outputFiles = [
        '/content/preprocessed_data1.h5',
        '/content/preprocessed_data2.h5'
    ]

    preprocessedFiles = []

    for zipPath, outputPath in zip(zipFiles, outputFiles):
        preprocessedFiles.append(outputPath)
        preprocessMedicalData(zipPath, metadataPath, outputPath)
        gc.collect()  # fro freeing up memory

    finalOutputPath = '/content/final_preprocessed_dataset.h5'
    finalSlices, finalLabels, _ = combinePreprocessedFiles(preprocessedFiles, finalOutputPath)