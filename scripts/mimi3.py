from pyhealth.datasets import MIMIC3Dataset

root_dir = "//scripts/data/mimic-3"

mimic3_ds = MIMIC3Dataset(
    root= root_dir,
    tables=['NOTEEVENTS', 'DIAGNOSES_ICD', 'D_ICD_DIAGNOSES','D_ICD_PRO CEDURES'],
)

print(mimic3_ds)