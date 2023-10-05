# Clinical
## TCGA
Download "clinical" from https://portal.gdc.cancer.gov/projects/TCGA-LUSC as a tsv file


### Slide label (tumor vs normal)

Real links for dataset (TCGA LUSC tissue slides):
* https://portal.gdc.cancer.gov/repository?filters={"op"%3A"and"%2C"content"%3A[{"content"%3A{"field"%3A"cases.case_id"%2C"value"%3A["set_id%3AQg2H84kBPtUv30ofTHho"]}%2C"op"%3A"IN"}%2C{"op"%3A"in"%2C"content"%3A{"field"%3A"files.data_format"%2C"value"%3A["svs"]}}%2C{"op"%3A"in"%2C"content"%3A{"field"%3A"files.experimental_strategy"%2C"value"%3A["Tissue Slide"]}}]}

If you add all those slides to the cart, you can then download a "sample sheet" in the "cart section" of TCGA.
To get normal vs LUSC slides:
```bash
cut -d$'\t' -f2,8 gdc_sample_sheet.2023-08-14.tsv | sed 's/\([0-9A-Z]\)\.[^ \t]\+/\1/ ; s/Solid Tissue // ; s/Primary // ; 1d' > sample_sheet_filtered.tsv 
grep Normal sample_sheet_filtered.tsv | cut -d$'\t' -f1 > normal_filtered.txt 
grep Tumor sample_sheet_filtered.tsv | cut -d$'\t' -f1 > tumor_filtered.txt 
# to get the files into separate directories
for file in /Data/TCGA_LUSC/preprocessed/TCGA/tiles/*; do command grep $(basename $file) tumor_filtered.txt && cp -r ${file} /Data/TCGA_LUSC/preprocessed/by
_class/lung_scc/ || cp -r ${file} /Data/TCGA_LUSC/preprocessed/by_class/lung_n; done
```

Supplmentary data:
Click on any patient here: https://portal.gdc.cancer.gov/exploration?cases_size=100&filters=%7B%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22cases.case_id%22%2C%22value%22%3A%5B%22set_id%3AvuFI_4oBYxjDXXXq0ln1%22%5D%7D%2C%22op%22%3A%22IN%22%7D%5D%2C%22op%22%3A%22AND%22%7D

And then find the "nationwidechildrens.org_clinical_follow_up_v1.0_lusc.txt"


## CPTAC
Annotations: download "clinical", then "clinical.tsv": [https://portal.gdc.cancer.gov/projects/CPTAC-3](https://portal.gdc.cancer.gov/projects/CPTAC-3) 