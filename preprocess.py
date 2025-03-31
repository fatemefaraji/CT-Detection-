import os
import glob
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import multiprocessing
from multiprocessing import Pool, cpu_count
import gc
import logging
import pylidc as pl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

class MedicalImagePreprocessor:
    def __init__(self, metadata_path, output_path, extract_dir="/content/drive/MyDrive/extractedData", skip_missing_labels=False):
        self.metadata_path = metadata_path
        self.output_path = output_path
        self.extract_dir = extract_dir
        self.processed_count = 0
        self.skipped_dicom_count = 0
        self.failed_studies = 0
        self.missing_annotations_count = 0
        self.skip_missing_labels = skip_missing_labels

    def get_annotations_with_pylidc(self):
        logging.info("Loading annotations with pylidc...")
        scans = pl.query(pl.Scan).all()
        logging.info(f"Found {len(scans)} scans with pylidc")

        annotations = {}
        for scan in scans:
            study_uid = scan.study_instance_uid
            # Get all nodules in the scan
            nodules = scan.annotations
            if not nodules:
                annotations[study_uid] = 0  # No nodules, assume benign
                continue
            
            # Get the maximum malignancy score from all nodules
            malignancy_scores = []
            for nodule in nodules:
                for ann in nodule:
                    malignancy = ann.malignancy  # 1 to 5 scale
                    if malignancy is not None:
                        malignancy_scores.append(malignancy)
            
            if malignancy_scores:
                annotations[study_uid] = max(malignancy_scores)
            else:
                annotations[study_uid] = 0  # No malignancy scores, assume benign

        logging.info(f"Extracted annotations for {len(annotations)} studies")
        sample_annotations = dict(list(annotations.items())[:5])
        logging.info(f"Sample annotations: {sample_annotations}")
        return annotations

    def process_dicom_file(self, dicom_path, study_uid, annotations):
        try:
            ds = pydicom.dcmread(dicom_path, force=True)
            
            if 'PixelData' not in ds:
                logging.warning(f"No pixel data in {dicom_path}, skipping")
                self.skipped_dicom_count += 1
                return None, None
            
            image = ds.pixel_array.astype(np.float32)
            
            original_shape = image.shape
            
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val == min_val:
                logging.warning(f"Image has constant value in {dicom_path}")
                image = np.zeros_like(image)
            else:
                image = (image - min_val) / (max_val - min_val)
            
            image = np.array(tf.image.resize(image[..., np.newaxis], (128, 128)))
            
            label = 1 if annotations.get(study_uid, 0) > 3 else 0
            
            if np.random.random() < 0.01:
                logging.info(f"Processed DICOM: {dicom_path}, Original shape: {original_shape}, "
                             f"New shape: {image.shape}, StudyUID: {study_uid}, Label: {label}")
            
            return image, label
        except Exception as e:
            logging.error(f"Error processing DICOM file {dicom_path}: {e}")
            try:
                ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                logging.debug(f"DICOM metadata for {dicom_path}: Rows={ds.get('Rows')}, Columns={ds.get('Columns')}, "
                              f"BitsAllocated={ds.get('BitsAllocated')}, TransferSyntaxUID={ds.get('TransferSyntaxUID')}")
            except:
                logging.debug(f"Could not read metadata for {dicom_path}")
            self.skipped_dicom_count += 1
            return None, None

    def preprocess_images(self):
        logging.info(f"Preprocessing images from metadata: {self.metadata_path}")
        
        if not os.path.exists(self.metadata_path):
            logging.error(f"Metadata file not found: {self.metadata_path}")
            return None, None, None
        
        try:
            metadata_df = pd.read_csv(self.metadata_path)
            logging.info(f"Metadata loaded, shape: {metadata_df.shape}")
            logging.info(f"Metadata columns: {metadata_df.columns.tolist()}")
            logging.info(f"Sample metadata rows: {metadata_df.head(5).to_dict('records')}")
            
            if 'Modality' in metadata_df.columns:
                modalities = metadata_df['Modality'].value_counts().to_dict()
                logging.info(f"Modality counts: {modalities}")
                ct_series = metadata_df[metadata_df['Modality'] == 'CT']
                logging.info(f"Found {len(ct_series)} CT series")
            else:
                logging.warning("Modality column not found, using all rows")
                ct_series = metadata_df
        except Exception as e:
            logging.error(f"Error reading metadata: {e}")
            return None, None, None

        # Get annotations using pylidc
        annotations = self.get_annotations_with_pylidc()

        slices, labels, study_uids = [], [], []
        processed_studies = 0
        total_studies = len(ct_series)

        for idx, row in ct_series.iterrows():
            processed_studies += 1
            if processed_studies % 10 == 0:
                logging.info(f"Processing study {processed_studies}/{total_studies}")
            
            if 'Study UID' not in row or 'File Location' not in row:
                logging.warning(f"Required columns missing in row: {row}")
                continue
                
            study_uid = row['Study UID']
            file_location = row['File Location']
            
            if not isinstance(file_location, str):
                logging.warning(f"Invalid file location for study {study_uid}: {file_location}")
                continue
            
            file_location = file_location.lstrip('.\\').lstrip('./').replace('\\', '/')
            logging.info(f"Processing Study UID: {study_uid}, File Location: {file_location}")
            
            if study_uid not in annotations:
                logging.warning(f"Study UID {study_uid} not found in annotations, defaulting label to 0")
                self.missing_annotations_count += 1
                if self.skip_missing_labels:
                    logging.info(f"Skipping study {study_uid} due to missing label (skip_missing_labels=True)")
                    continue
            
            # Use pylidc to find the scan
            scan = pl.query(pl.Scan).filter(pl.Scan.study_instance_uid == study_uid).first()
            if not scan:
                logging.warning(f"No scan found for Study UID {study_uid} using pylidc")
                self.failed_studies += 1
                continue

            dicom_files = scan.get_path_to_dicom_files()
            dicom_files = glob.glob(os.path.join(dicom_files, "*.dcm"))
            logging.info(f"Found {len(dicom_files)} DICOM files for study {study_uid} using pylidc")
            if dicom_files:
                logging.info(f"Sample DICOM files: {dicom_files[:5]}")
            else:
                logging.warning(f"No DICOM files found for study {study_uid} using pylidc")
                self.failed_studies += 1
                continue
            
            with Pool(max(1, cpu_count() // 2)) as pool:
                results = pool.starmap(self.process_dicom_file,
                                      [(dicom_file, study_uid, annotations) for dicom_file in dicom_files])
            
            batch_success = 0
            for image, label in results:
                if image is not None and label is not None:
                    slices.append(image)
                    labels.append(label)
                    study_uids.append(study_uid)
                    batch_success += 1
                    self.processed_count += 1
            
            logging.info(f"Successfully processed {batch_success}/{len(dicom_files)} images for study {study_uid}")

        logging.info(f"Total processed images: {len(slices)}")
        logging.info(f"Total studies processed: {processed_studies - self.failed_studies}/{total_studies}")
        logging.info(f"Total studies failed (no DICOM files): {self.failed_studies}")
        logging.info(f"Total studies missing annotations: {self.missing_annotations_count}")
        logging.info(f"Total skipped DICOM files due to errors: {self.skipped_dicom_count}")
        if len(slices) > 0:
            logging.info(f"First image shape: {slices[0].shape}, dtype: {slices[0].dtype}")
            logging.info(f"Labels distribution: {np.bincount(labels)}")
        else:
            logging.error("No images were processed successfully!")
            return None, None, None

        slices = np.array(slices)
        labels = np.array(labels)
        
        logging.info(f"Final dataset - Slices shape: {slices.shape}, Labels shape: {labels.shape}")
        logging.info(f"Memory usage - Slices: {slices.nbytes/1024/1024:.2f} MB")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        try:
            with h5py.File(self.output_path, 'w') as hdf:
                hdf.create_dataset('slices', data=slices, compression='gzip', chunks=True)
                hdf.create_dataset('labels', data=labels, compression='gzip')
                hdf.create_dataset('studyUids', data=np.array(study_uids, dtype='S'), compression='gzip')
            
            file_size = os.path.getsize(self.output_path) / 1024
            logging.info(f"Output file created: {self.output_path}, Size: {file_size:.2f} KB")
        except Exception as e:
            logging.error(f"Error saving H5 file: {e}")

        logging.info("Image preprocessing complete.")
        return slices, labels, study_uids

    def process(self):
        return self.preprocess_images()

if __name__ == "__main__":
    # Install pylidc
    !pip install pylidc

    # Create the .pylidcrc configuration file
    import os
    pylidc_config = """
    [paths]
    dicom = /content/extractedData/
    xml = /content/drive/MyDrive/xml_annotations/
    """
    with open(os.path.expanduser("~/.pylidcrc"), "w") as f:
        f.write(pylidc_config)

    # Unzip the XML annotations
    !unzip /content/LIDC-XML-only.zip -d /content/extractedData/xml_annotations

    metadata_path = '/content/drive/MyDrive/metadata.csv'
    output_path = '/content/final_preprocessed_data.h1'
    
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file not found: {metadata_path}")
        exit(1)
    
    logging.info("Starting preprocessing...")
    preprocessor = MedicalImagePreprocessor(metadata_path, output_path, skip_missing_labels=False)
    slices, labels, study_uids = preprocessor.process()
    
    if slices is not None:
        logging.info("Preprocessing completed successfully.")
    else:
        logging.error("Preprocessing failed!")
