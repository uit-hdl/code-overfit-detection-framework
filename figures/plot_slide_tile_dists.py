# -*- coding: utf-8 -*-
'''
This code will re-create figure X and Y in the paper "Open-source framework for detecting bias and overfitting for large pathology images
'''
import numpy as np
from bokeh.layouts import layout
from bokeh.models import CustomJSTickFormatter
from bokeh.plotting import save, figure, output_file

data = """156 66
100 60
97 22
76 77
70 85
68 18
60 33
57 56
57 34
55 43
51 39
34 21
30 63
24 58
23 37
17 98
16 NC
16 90
15 51
14 46
10 94
8 68
8 52
7 96
7 92
6 O2
5 NK
4 70
2 LA
2 L3
1 XC
1 MF
1 J1
1 79
1 6A"""



slide_data = """49355 85
32686 39
25694 66
19499 22
19330 60
15870 34
13515 33
10991 43
10278 77
9917 21
8369 56
7229 37
5661 63
5278 18
3950 98
3270 58
2947 70
2732 90
2425 46
2177 92
1912 NC
1848 94
1484 O2
1400 96
1320 51
1202 68
1094 52
589 NK
534 L3
374 LA
253 XC
249 6A
125 MF
52 J1"""

institution_lookup = {
"01": "International Genomics Consortium",
"02": "MD Anderson Cancer Center",
"04": "Gynecologic Oncology Group",
"05": "Indivumed",
"06": "Henry Ford Hospital",
"07": "TGen",
"08": "UCSF",
"09": "UCSF",
"10": "MD Anderson Cancer Center",
"11": "MD Anderson Cancer Center",
"12": "Duke",
"13": "Memorial Sloan Kettering",
"14": "Emory University",
"15": "Mayo Clinic - Rochester",
"16": "Toronto Western Hospital",
"17": "Washington University",
"18": "Princess Margaret Hospital (Canada)",
"19": "Case Western",
"1Z": "Johns Hopkins",
"20": "Fox Chase Cancer Center",
"21": "Fox Chase Cancer Center",
"22": "Mayo Clinic - Rochester",
"23": "Cedars Sinai",
"24": "Washington University",
"25": "Mayo Clinic - Rochester",
"26": "University of Florida",
"27": "Milan - Italy, Fondazione IRCCS Instituto Neuroligico C. Besta",
"28": "Cedars Sinai",
"29": "Duke",
"2A": "Memorial Sloan Kettering Cancer Center",
"2E": "University of Kansas Medical Center",
"2F": "Erasmus MC",
"2G": "Erasmus MC",
"2H": "Erasmus MC",
"2J": "Mayo Clinic",
"2K": "Greenville Health System",
"2L": "Technical University of Munich",
"2M": "Technical University of Munich",
"2N": "Technical University of Munich",
"2P": "University of California San Diego",
"2V": "University of California San Diego",
"2W": "University of New Mexico",
"2X": "ABS IUPUI",
"2Y": "Moffitt Cancer Center",
"2Z": "Moffitt Cancer Center",
"30": "Harvard",
"31": "Imperial College",
"32": "St. Joseph's Hospital (AZ)",
"33": "Johns Hopkins",
"34": "University of Pittsburgh",
"35": "Cureline",
"36": "BC Cancer Agency",
"37": "Cureline",
"38": "UNC",
"39": "MSKCC",
"3A": "Moffitt Cancer Center",
"3B": "Moffitt Cancer Center",
"3C": "Columbia University",
"3E": "Columbia University",
"3G": "MD Anderson Cancer Center",
"3H": "MD Anderson Cancer Center",
"3J": "Carle Cancer Center",
"3K": "Boston Medical Center",
"3L": "Albert Einstein Medical Center",
"3M": "University of Kansas Medical Center",
"3N": "Greenville Health System",
"3P": "Greenville Health System",
"3Q": "Greenville Health Systems",
"3R": "University of New Mexico",
"3S": "University of New Mexico",
"3T": "Emory University",
"3U": "University of Chicago",
"3W": "University of California San Diego",
"3X": "Alberta Health Services",
"3Z": "Mary Bird Perkins Cancer Center - Our Lady of the Lake",
"41": "Christiana Healthcare",
"42": "Christiana Healthcare",
"43": "Christiana Healthcare",
"44": "Christiana Healthcare",
"46": "St. Joseph's Medical Center (MD)",
"49": "Johns Hopkins",
"4A": "Mary Bird Perkins Cancer Center - Our Lady of the Lake",
"4B": "Mary Bird Perkins Cancer Center - Our Lady of the Lake",
"4C": "Mary Bird Perkins Cancer Center - Our Lady of the Lake",
"4D": "Molecular Response",
"4E": "Molecular Response",
"4G": "Sapienza University of Rome",
"4H": "Proteogenex, Inc.",
"4J": "Proteogenex, Inc.",
"4K": "Proteogenex, Inc.",
"4L": "Proteogenex, Inc.",
"4N": "Mary Bird Perkins Cancer Center - Our Lady of the Lake",
"4P": "Duke University",
"4Q": "Duke University",
"4R": "Duke University",
"4S": "Duke University",
"4T": "Duke University",
"4V": "Hospital Louis Pradel",
"4W": "University of Miami",
"4X": "Yale University",
"4Y": "Medical College of Wisconsin",
"4Z": "Barretos Cancer Hospital",
"50": "University of Pittsburgh",
"51": "UNC",
"52": "University of Miami",
"53": "University of Miami",
"55": "International Genomics Consortium",
"56": "International Genomics Consortium",
"57": "International Genomics Consortium",
"58": "Thoraxklinik at University Hospital Heidelberg",
"59": "Roswell Park",
"5A": "Wake Forest University",
"5B": "Medical College of Wisconsin",
"5C": "Cureline",
"5D": "University of Miami",
"5F": "Duke University",
"5G": "Cleveland Clinic Foundation",
"5H": "Retina Consultants Houston",
"5J": "Cureline",
"5K": "St. Joseph's Hospital AZ",
"5L": "University of Sao Paulo",
"5M": "University of Sao Paulo",
"5N": "University Hospital Erlangen",
"5P": "University Hospital Erlangen",
"5Q": "Proteogenex, Inc",
"5R": "Proteogenex, Inc",
"5S": "Holy Cross",
"5T": "Holy Cross",
"5U": "Regina Elena National Cancer Institute",
"5V": "Roswell Park",
"5W": "University of Alabama",
"5X": "University of Alabama",
"60": "Roswell Park",
"61": "University of Pittsburgh",
"62": "Thoraxklinik at University Hospital Heidelberg",
"63": "Ontario Institute for Cancer Research",
"64": "Fox Chase",
"65": "Roswell Park",
"66": "Indivumed",
"67": "St Joseph's Medical Center (MD)",
"68": "Washington University - Cleveland Clinic",
"69": "Washington University - Cleveland Clinic",
"6A": "University of Kansas",
"6D": "University of Oklahoma HSC",
"6G": "University of Sao Paulo",
"6H": "Test For lcml",
"70": "ILSBio",
"71": "ILSBio",
"72": "NCH",
"73": "Roswell Park",
"74": "Swedish Neurosciences",
"75": "Ontario Institute for Cancer Research (OICR)",
"76": "Thomas Jefferson University",
"77": "Prince Charles Hospital",
"78": "Prince Charles Hospital",
"79": "Ontario Institute for Cancer Research (OICR)/Ottawa",
"80": "Ontario Institute for Cancer Research (OICR)/Ottawa",
"81": "CHI-Penrose Colorado",
"82": "CHI-Penrose Colorado",
"83": "CHI-Penrose Colorado",
"85": "Asterand",
"86": "Asterand",
"87": "International Genomics Consortium",
"90": "ABS - IUPUI",
"91": "ABS - IUPUI",
"92": "Washington University - St. Louis",
"93": "Washington University - St. Louis",
"94": "Washington University - Emory",
"95": "Washington University - Emory",
"96": "Washington University - NYU",
"97": "Washington University - NYU",
"98": "Washington University - Alabama",
"99": "Washington University - Alabama",
"A1": "UCSF",
"A2": "Walter Reed",
"A3": "International Genomics Consortium",
"A4": "International Genomics Consortium",
"A5": "Cedars Sinai",
"A6": "Christiana Healthcare",
"A7": "Christiana Healthcare",
"A8": "Indivumed",
"AA": "Indivumed",
"AB": "Washington University",
"AC": "International Genomics Consortium",
"AD": "International Genomics Consortium",
"AF": "Christiana Healthcare",
"AG": "Indivumed",
"AH": "International Genomics Consortium",
"AJ": "International Genomics Conosrtium",
"AK": "Fox Chase",
"AL": "Fox Chase",
"AM": "Cureline",
"AN": "Cureline",
"AO": "MSKCC",
"AP": "MSKCC",
"AQ": "UNC",
"AR": "Mayo",
"AS": "St. Joseph's Medical Center-(MD)",
"AT": "St. Joseph's Medical Center-(MD)",
"AU": "St. Joseph's Medical Center-(MD)",
"AV": "NCH",
"AW": "Cureline",
"AX": "Gynecologic Oncology Group",
"AY": "UNC",
"AZ": "University of Pittsburgh",
"B0": "University of Pittsburgh",
"B1": "University of Pittsburgh",
"B2": "Christiana Healthcare",
"B3": "Christiana Healthcare",
"B4": "Cureline",
"B5": "Duke",
"B6": "Duke",
"B7": "Cureline",
"B8": "UNC",
"B9": "UNC",
"BA": "UNC",
"BB": "Johns Hopkins",
"BC": "UNC",
"BD": "University of Pittsburgh",
"BF": "Cureline",
"BG": "University of Pittsburgh",
"BH": "University of Pittsburgh",
"BI": "University of Pittsburgh",
"BJ": "University of Pittsburgh",
"BK": "Christiana Healthcare",
"BL": "Christiana Healthcare",
"BM": "UNC",
"BP": "MSKCC",
"BQ": "MSKCC",
"BR": "Asterand",
"BS": "University of Hawaii",
"BT": "University of Pittsburgh",
"BW": "St. Joseph's Medical Center-(MD)",
"C4": "Indivumed",
"C5": "Medical College of Wisconsin",
"C8": "ILSBio",
"C9": "ILSBio",
"CA": "ILSBio",
"CB": "ILSBio",
"CC": "ILSBio",
"CD": "ILSBio",
"CE": "ILSBio",
"CF": "ILSBio",
"CG": "Indivumed",
"CH": "Indivumed",
"CI": "University of Pittsburgh",
"CJ": "MD Anderson Cancer Center",
"CK": "Harvard",
"CL": "Harvard",
"CM": "MSKCC",
"CN": "University of Pittsburgh",
"CQ": "University Health Network, Toronto",
"CR": "Vanderbilt University",
"CS": "Thomas Jefferson University",
"CU": "UNC",
"CV": "MD Anderson Cancer Center",
"CW": "Mayo Clinic - Rochester",
"CX": "Medical College of Georgia",
"CZ": "Harvard",
"D1": "Mayo Clinic",
"D3": "MD Anderson",
"D5": "Greater Poland Cancer Center",
"D6": "Greater Poland Cancer Center",
"D7": "Greater Poland Cancer Center",
"D8": "Greater Poland Cancer Center",
"D9": "Greater Poland Cancer Center",
"DA": "Yale",
"DB": "Mayo Clinic - Rochester",
"DC": "MSKCC",
"DD": "Mayo Clinic - Rochester",
"DE": "University of North Carolina",
"DF": "Ontario Institute for Cancer Research",
"DG": "Ontario Institute for Cancer Research",
"DH": "University of Florida",
"DI": "MD Anderson",
"DJ": "Memorial Sloan Kettering",
"DK": "Memorial Sloan Kettering",
"DM": "University Of Michigan",
"DO": "Medical College of Georgia",
"DQ": "University Of Michigan",
"DR": "University of Hawaii",
"DS": "Cedars Sinai",
"DT": "ILSBio",
"DU": "Henry Ford Hospital",
"DV": "NCI Urologic Oncology Branch",
"DW": "NCI Urologic Oncology Branch",
"DX": "Memorial Sloan Kettering",
"DY": "University Of Michigan",
"DZ": "Mayo Clinic - Rochester",
"E1": "Duke",
"E2": "Roswell Park",
"E3": "Roswell Park",
"E5": "Roswell Park",
"E6": "Roswell Park",
"E7": "Asterand",
"E8": "Asterand",
"E9": "Asterand",
"EA": "Asterand",
"EB": "Asterand",
"EC": "Asterand",
"ED": "Asterand",
"EE": "University of Sydney",
"EF": "Cureline",
"EI": "Greater Poland Cancer Center",
"EJ": "University of Pittsburgh",
"EK": "Gynecologic Oncology Group",
"EL": "MD Anderson",
"EM": "University Health Network",
"EO": "University Health Network",
"EP": "Christiana Healthcare",
"EQ": "Christiana Healthcare",
"ER": "University of Pittsburgh",
"ES": "University of Florida",
"ET": "Johns Hopkins",
"EU": "CHI-Penrose Colorado",
"EV": "CHI-Penrose Colorado",
"EW": "University of Miami",
"EX": "University of North Carolina",
"EY": "University of North Carolina",
"EZ": "UNC",
"F1": "UNC",
"F2": "UNC",
"F4": "Asterand",
"F5": "Asterand",
"F6": "Asterand",
"F7": "Asterand",
"F9": "Asterand",
"FA": "Asterand",
"FB": "Asterand",
"FC": "Asterand",
"FD": "BLN - University Of Chicago",
"FE": "Ohio State University",
"FF": "SingHealth",
"FG": "Case Western",
"FH": "CHI-Penrose Colorado",
"FI": "Washington University",
"FJ": "BLN - Baylor",
"FK": "Johns Hopkins",
"FL": "University of Hawaii - Normal Study",
"FM": "International Genomics Consortium",
"FN": "International Genomics Consortium",
"FP": "International Genomics Consortium",
"FQ": "Johns Hopkins",
"FR": "University of North Carolina",
"FS": "Essen",
"FT": "BLN - University of Miami",
"FU": "International Genomics Consortium",
"FV": "International Genomics Consortium",
"FW": "International Genomics Consortium",
"FX": "International Genomics Consortium",
"FY": "International Genomics Consortium",
"FZ": "University of Pittsburgh",
"G2": "MD Anderson",
"G3": "Alberta Health Services",
"G4": "Roswell Park",
"G5": "Roswell Park",
"G6": "Roswell Park",
"G7": "Roswell Park",
"G8": "Roswell Park",
"G9": "Roswell Park",
"GC": "International Genomics Consortium",
"GD": "ABS - IUPUI",
"GE": "ABS - IUPUI",
"GF": "ABS - IUPUI",
"GG": "ABS - IUPUI",
"GH": "ABS - IUPUI",
"GI": "ABS - IUPUI",
"GJ": "ABS - IUPUI",
"GK": "ABS - IUPUI",
"GL": "ABS - IUPUI",
"GM": "MD Anderson",
"GN": "Roswell",
"GP": "MD Anderson",
"GR": "University of Nebraska Medical Center (UNMC)",
"GS": "Fundacio Clinic per a la Recerca Biomedica",
"GU": "BLN - UT Southwestern Medical Center at Dallas",
"GV": "BLN - Cleveland Clinic",
"GZ": "BC Cancer Agency",
"H1": "Medical College of Georgia",
"H2": "Christiana Healthcare",
"H3": "ABS - IUPUI",
"H4": "Medical College of Georgia",
"H5": "Medical College of Georgia",
"H6": "Christiana Healthcare",
"H7": "ABS - IUPUI",
"H8": "ABS - IUPUI",
"H9": "ABS - IUPUI",
"HA": "Alberta Health Services",
"HB": "University of North Carolina",
"HC": "International Genomics Consortium",
"HD": "International Genomics Consortium",
"HE": "Ontario Institute for Cancer Research (OICR)",
"HF": "Ontario Institute for Cancer Research (OICR)",
"HG": "Roswell Park",
"HH": "Fox Chase",
"HI": "Fox Chase",
"HJ": "Fox Chase",
"HK": "Fox Chase",
"HL": "Fox Chase",
"HM": "Christiana Healthcare",
"HN": "Ontario Institute for Cancer Research (OICR)",
"HP": "Ontario Institute for Cancer Research (OICR)",
"HQ": "Ontario Institute for Cancer Research (OICR)",
"HR": "Ontario Institute for Cancer Research (OICR)",
"HS": "Ontario Institute for Cancer Research (OICR)",
"HT": "Case Western - St Joes",
"HU": "National Cancer Center Korea",
"HV": "National Cancer Center Korea",
"HW": "MSKCC",
"HZ": "International Genomics Consortium",
"IA": "Cleveland Clinic",
"IB": "Alberta Health Services",
"IC": "University of Pittsburgh",
"IE": "ABS - IUPUI",
"IF": "University of Texas MD Anderson Cancer Center",
"IG": "Asterand",
"IH": "University of Miami",
"IJ": "Christiana Healthcare",
"IK": "Christiana Healthcare",
"IM": "University of Miami",
"IN": "University of Pittsburgh",
"IP": "ABS - IUPUI",
"IQ": "University of Miami",
"IR": "Memorial Sloan Kettering",
"IS": "Memorial Sloan Kettering",
"IW": "Cedars Sinai",
"IZ": "ABS - Lahey Clinic",
"J1": "ABS - Lahey Clinic",
"J2": "ABS - Lahey Clinic",
"J4": "ABS - Lahey Clinic",
"J7": "ILSBio",
"J8": "Mayo Clinic",
"J9": "Melbourne Health",
"JA": "ABS - Research Metrics Pakistan",
"JL": "ABS - Research Metrics Pakistan",
"JU": "BLN - Baylor",
"JV": "BLN - Baylor",
"JW": "BLN - Baylor",
"JX": "Washington University",
"JY": "University Health Network",
"JZ": "University of Rochester",
"K1": "University of Pittsburgh",
"K4": "ABS - Lahey Clinic",
"K6": "ABS - Lahey Clinic",
"K7": "ABS - Lahey Clinic",
"K8": "ABS - Lahey Clinic",
"KA": "ABS - Lahey Clinic",
"KB": "University Health Network, Toronto",
"KC": "Cornell Medical College",
"KD": "Mount Sinai School of Medicine",
"KE": "Mount Sinai School of Medicine",
"KF": "Christiana Healthcare",
"KG": "Baylor Network",
"KH": "Memorial Sloan Kettering",
"KJ": "University of Miami",
"KK": "MD Anderson Cancer Center",
"KL": "MSKCC",
"KM": "NCI Urologic Oncology Branch",
"KN": "Harvard",
"KO": "MD Anderson Cancer Center",
"KP": "British Columbia Cancer Agency",
"KQ": "Cornell Medical College",
"KR": "University Of Michigan",
"KS": "University Of Michigan",
"KT": "Hartford",
"KU": "Hartford",
"KV": "Hartford",
"KZ": "Hartford",
"L1": "Hartford",
"L3": "Gundersen Lutheran Health System",
"L4": "Gundersen Lutheran Health System",
"L5": "University of Michigan",
"L6": "National Institutes of Health",
"L7": "Christiana Care",
"L8": "University of Miami",
"L9": "Candler",
"LA": "Candler",
"LB": "Candler",
"LC": "Hartford Hospital",
"LD": "Hartford Hospital",
"LG": "Hartford Hospital",
"LH": "Hartford Hospital",
"LI": "Hartford Hospital",
"LK": "University of Pittsburgh",
"LL": "Candler",
"LN": "ILSBIO",
"LP": "ILSBIO",
"LQ": "Gundersen Lutheran Health System",
"LS": "Gundersen Lutheran Health System",
"LT": "Gundersen Lutheran Health System",
"M7": "University of North Carolina",
"M8": "Ontario Institute for Cancer Research (OICR)",
"M9": "Ontario Institute for Cancer Research (OICR)",
"MA": "MD Anderson Cancer Center",
"MB": "University of Minnesota",
"ME": "University of Minnesota",
"MF": "University of Minnesota",
"MG": "BLN - Baylor",
"MH": "BLN - Baylor",
"MI": "BLN - Baylor",
"MJ": "BLN - Baylor",
"MK": "BLN - Baylor",
"ML": "BLN - Baylor",
"MM": "BLN - Baylor",
"MN": "BLN - Baylor",
"MO": "ILSBio",
"MP": "Washington University - Mayo Clinic",
"MQ": "Washington University - NYU",
"MR": "University of Minnesota",
"MS": "University of Minnesota",
"MT": "University of Minnesota",
"MU": "University of Minnesota",
"MV": "University of Minnesota",
"MW": "University of Miami",
"MX": "MSKCC",
"MY": "Montefiore Medical Center",
"MZ": "Montefiore Medical Center",
"N1": "Montefiore Medical Center",
"N5": "MSKCC",
"N6": "University of Pittsburgh",
"N7": "Washington University",
"N8": "University of North Carolina",
"N9": "MD Anderson",
"NA": "Duke University",
"NB": "Washington University - CHUV",
"NC": "Washington University - CHUV",
"ND": "Cedars Sinai",
"NF": "Mayo Clinic - Rochester",
"NG": "Roswell Park",
"NH": "Candler",
"NI": "Roswell Park",
"NJ": "Washington University - Rush University",
"NK": "Washington University - Rush University",
"NM": "Cambridge BioSource",
"NP": "International Genomics Consortium",
"NQ": "International Genomics Consortium",
"NS": "Gundersen Lutheran Health System",
"O1": "Washington University - CALGB",
"O2": "Washington University - CALGB",
"O8": "Saint Mary's Health Care",
"O9": "Saint Mary's Health Care",
"OC": "Saint Mary's Health Care",
"OD": "Saint Mary's Health Care",
"OE": "Saint Mary's Health Care",
"OJ": "Saint Mary's Health Care",
"OK": "Mount Sinai School of Medicine",
"OL": "University of Chicago",
"OR": "University of Michigan",
"OU": "Roswell Park",
"OW": "International Genomics Consortium",
"OX": "University of North Carolina",
"OY": "University of North Carolina",
"P3": "Fred Hutchinson",
"P4": "MD Anderson Cancer Center",
"P5": "Cureline",
"P6": "Translational Genomics Research Institute",
"P7": "Translational Genomics Research Institute",
"P8": "University of Pittsburgh",
"P9": "University of Minnesota",
"PA": "University of Minnesota",
"PB": "University of Minnesota",
"PC": "Fox Chase",
"PD": "Fox Chase",
"PE": "Fox Chase",
"PG": "Montefiore Medical Center",
"PH": "Gundersen Lutheran",
"PJ": "Gundersen Lutheran",
"PK": "University Health Network",
"PL": "Institute of Human Virology Nigeria",
"PN": "Institute of Human Virology Nigeria",
"PQ": "University of Colorado Denver",
"PR": "Roswell Park",
"PT": "Maine Medical Center",
"PZ": "ABS - Lahey Clinic",
"Q1": "University of Oklahoma HSC",
"Q2": "University of Oklahoma HSC",
"Q3": "University of Oklahoma HSC",
"Q4": "Emory University",
"Q9": "Emory University",
"QA": "Emory University",
"QB": "Emory University",
"QC": "Emory University",
"QD": "Emory University",
"QF": "BLN - Baylor",
"QG": "BLN - Baylor",
"QH": "Fondazione-Besta",
"QJ": "Mount Sinai School of Medicine",
"QK": "Emory University - Winship Cancer Inst.",
"QL": "University of Chicago",
"QM": "University of Oklahoma HSC",
"QN": "ILSBio",
"QQ": "Roswell Park",
"QR": "National Institutes of Health",
"QS": "Candler",
"QT": "University of North Carolina",
"QU": "Harvard Beth Israel",
"QV": "Instituto Nacional de Cancerologia",
"QW": "Instituto Nacional de Cancerologia",
"R1": "CHI-Penrose Colorado",
"R2": "CHI-Penrose Colorado",
"R3": "CHI-Penrose Colorado",
"R5": "MD Anderson Cancer Center",
"R6": "MD Anderson Cancer Center",
"R7": "Gundersen Lutheran Health System",
"R8": "MD Anderson",
"R9": "Candler",
"RA": "Candler",
"RB": "Emory University",
"RC": "University of Utah",
"RD": "Peter MacCallum Cancer Center",
"RE": "Peter MacCallum Cancer Center",
"RG": "Montefiore Medical Center",
"RH": "BLN - Baylor",
"RL": "St. Joseph's Hospital AZ",
"RM": "St. Joseph's Hospital AZ",
"RN": "St. Joseph's Hospital AZ",
"RP": "St. Joseph's Hospital AZ",
"RQ": "St. Joseph's Hospital AZ",
"RR": "St. Joseph's Hospital AZ",
"RS": "Memorial Sloan Kettering Cancer Center",
"RT": "Cleveland Clinic Foundation",
"RU": "Northwestern University",
"RV": "Northwestern University",
"RW": "Michigan University",
"RX": "University of Minnesota",
"RY": "University of California San Francisco",
"RZ": "Wills Eye Institute",
"S2": "Albert Einstein Medical Center",
"S3": "Albert Einstein Medical Center",
"S4": "University of Chicago",
"S5": "University of Oklahoma HSC",
"S6": "Gundersen Lutheran Health System",
"S7": "University Hospital Motol",
"S8": "ABS - IUPUI",
"S9": "Dept of Neurosurgery at University of Heidelberg",
"SA": "ABS - IUPUI",
"SB": "Baylor College of Medicine",
"SC": "Memorial Sloan Kettering",
"SD": "MD Anderson",
"SE": "Boston Medical Center",
"SG": "Cleveland Clinic Foundation",
"SH": "Papworth Hospital",
"SI": "Washington University St. Louis",
"SJ": "Albert Einstein Medical Center",
"SK": "St. Joseph's Hospital AZ",
"SL": "St. Joseph's Hospital AZ",
"SN": "BLN - Baylor",
"SO": "University of Minnesota",
"SP": "University Health Network",
"SQ": "International Genomics Consortium",
"SR": "Tufts Medical Center",
"SS": "Medical College of Georgia",
"ST": "Global Bioclinical-Moldova",
"SU": "Global Bioclinical-Moldova",
"SW": "Global Bioclinical-Moldova",
"SX": "Mayo Clinic Arizona",
"SY": "Mayo Clinic Arizona",
"T1": "St. Joseph's Hospital Arizona",
"T2": "St. University of Colorado Denver",
"T3": "Molecular Response",
"T6": "Molecular Response",
"T7": "Molecular Response",
"T9": "Molecular Response",
"TE": "Global BioClinical - Georgia",
"TG": "Global BioClinical - Georgia",
"TK": "Global BioClinical - Georgia",
"TL": "Global BioClinical - Georgia",
"TM": "The University of New South Wales",
"TN": "Ohio State University",
"TP": "Maine Medical Center",
"TQ": "University of Sao Paulo",
"TR": "Global Bioclinical-Moldova",
"TS": "University of Pennsylvania",
"TT": "University of Pennsylvania",
"TV": "Wake Forest University",
"UB": "UCSF",
"UC": "University of Washington",
"UD": "University of Western Australia",
"UE": "Asterand",
"UF": "Barretos Cancer Hospital",
"UJ": "Boston Medical Center",
"UL": "Boston Medical Center",
"UN": "Boston Medical Center",
"UP": "Boston Medical Center",
"UR": "Boston Medical Center",
"US": "Garvan Institute of Medical Research",
"UT": "Asbestos Diseases Research Institute",
"UU": "Mary Bird Perkins Cancer Center - Our Lady of the Lake",
"UV": "Capital Biosciences",
"UW": "University of North Carolina",
"UY": "University of California San Francisco",
"UZ": "University of California San Francisco",
"V1": "University of California San Francisco",
"V2": "Cleveland Clinic Foundation",
"V3": "Cleveland Clinic Foundation",
"V4": "Institut Curie",
"V5": "Duke University",
"V6": "Duke University",
"V7": "Medical College of Georgia",
"V8": "Medical College of Georgia",
"V9": "Medical College of Georgia",
"VA": "Alliance",
"VB": "Global BioClinical - Georgia",
"VD": "University of Liverpool",
"VF": "University of Pennsylvania",
"VG": "Institute of Human Virology Nigeria",
"VK": "Institute of Human Virology Nigeria",
"VL": "Institute of Human Virology Nigeria",
"VM": "Huntsman Cancer Institute",
"VN": "NCI Urologic Oncology Branch",
"VP": "Washington University",
"VQ": "Barretos Cancer Hospital",
"VR": "Barretos Cancer Hospital",
"VS": "Barretos Cancer Hospital",
"VT": "Vanderbilt",
"VV": "John Wayne Cancer Center",
"VW": "Northwestern University",
"VX": "Northwestern University",
"VZ": "Albert Einstein Medical Center",
"W2": "Medical College of Wisconsin",
"W3": "John Wayne Cancer Center",
"W4": "University of North Carolina",
"W5": "Mayo Clinic Rochester",
"W6": "UCSF",
"W7": "Garvan Institute of Medical Research",
"W8": "Greenville Health System",
"W9": "University of Kansas",
"WA": "University of Schleswig-Holstein",
"WB": "Erasmus MC",
"WC": "MD Anderson",
"WD": "Emory University",
"WE": "Norfolk and Norwich Hospital",
"WF": "Greenville Health System",
"WG": "Greenville Health System",
"WH": "Greenville Health System",
"WJ": "Greenville Health System",
"WK": "Brigham and Women's Hospital",
"WL": "University of Kansas",
"WM": "University of Kansas",
"WN": "University of Kansas",
"WP": "University of Kansas",
"WQ": "University of Kansas",
"WR": "University of Kansas",
"WS": "University of Kansas",
"WT": "University of Kansas",
"WU": "Wake Forest University",
"WW": "Wake Forest University",
"WX": "Yale University",
"WY": "Johns Hopkins",
"WZ": "International Genomics Consortium",
"X2": "University of Washington",
"X3": "Cleveland Clinic Foundation",
"X4": "Institute for Medical Research",
"X5": "Institute of Human Virology Nigeria",
"X6": "University of Iowa",
"X7": "ABS IUPUI",
"X8": "St. Joseph's Hospital Arizona",
"X9": "University of California, Davis",
"XA": "University of Minnesota",
"XB": "Albert Einstein Medical Center",
"XC": "Albert Einstein Medical Center",
"XD": "Providence Portland Medical Center",
"XE": "University of Southern California",
"XF": "University of Southern California",
"XG": "BLN UT Southwestern Medical Center at Dallas",
"XH": "BLN Baylor",
"XJ": "University of Kansas",
"XK": "Mayo Clinic Arizona",
"XM": "MSKCC",
"XN": "University of Sao Paulo",
"XP": "University of Sao Paulo",
"XQ": "University of Sao Paulo",
"XR": "University of Sao Paulo",
"XS": "University of Sao Paulo",
"XT": "Johns Hopkins",
"XU": "University Health Network",
"XV": "Capital Biosciences",
"XX": "Spectrum Health",
"XY": "Spectrum Health",
"Y3": "University of New Mexico",
"Y5": "University of Arizona",
"Y6": "University of Arizona",
"Y8": "Spectrum Health",
"YA": "Spectrum Health",
"YB": "Spectrum Health",
"YC": "Spectrum Health",
"YD": "Spectrum Health",
"YF": "University of Puerto Rico",
"YG": "University of Puerto Rico",
"YH": "Stanford University",
"YJ": "Stanford University",
"YL": "PROCURE Biobank",
"YN": "University of Arizona",
"YR": "Barretos Cancer Hospital",
"YS": "Barretos Cancer Hospital",
"YT": "Barretos Cancer Hospital",
"YU": "Barretos Cancer Hospital",
"YV": "MSKCC",
"YW": "Albert Einstein Medical Center",
"YX": "Emory University",
"YY": "Roswell Park",
"YZ": "The Ohio State University",
"Z2": "IDI-IRCCS",
"Z3": "UCLA",
"Z4": "Cureline",
"Z5": "Cureline",
"Z6": "Cureline",
"Z7": "John Wayne Cancer Center",
"Z8": "John Wayne Cancer Center",
"ZA": "Candler",
"ZB": "Thoraxklinik",
"ZC": "University of Mannheim",
"ZD": "ILSbio",
"ZE": "Spectrum Health",
"ZF": "University of Sheffield",
"ZG": "University Medical Center Hamburg-Eppendorf",
"ZH": "University of North Carolina",
"ZJ": "NCI HRE Branch",
"ZK": "University of New Mexico",
"ZL": "Valley Hospital",
"ZM": "University of Ulm",
"ZN": "Brigham and Women's Hospital Division of Thoracic Surgery",
"ZP": "Medical College of Wisconsin",
"ZQ": "Tayside Tissue Bank",
"ZR": "Tayside Tissue Bank",
"ZS": "Tayside Tissue Bank",
"ZT": "International Genomics Consortium",
"ZU": "Spectrum Health",
"ZW": "University of Alabama",
"ZX": "University of Alabama",
}

# for inst in $(egrep -o "TCGA-.{2}" /tmp/a | sed 's/TCGA-//' | sort | uniq); do echo -n "$inst "; egrep "TCGA-$inst" /tmp/a | cut -d' ' -f1 | command paste -sd+ | bc;  done | sort -k2 -n -r
file_data = """85 51117491520
39 15966219057
66 14788446666
60 11825285922
22 10731198007
43 9453050623
34 9181833187
33 8506729006
77 7691365176
21 6565305568
56 6394610684
63 5604613560
58 5251705762
98 5142978817
37 4188176915
18 3578723746
NC 3238230608
90 2875555982
92 2077446131
O2 1777062090
94 1720031480
52 1566273126
70 1403472106
46 1330393864
96 1042197489
NK 902421977
51 884452955
68 821520000
L3 662144590
LA 476958754
XC 315494607
6A 254700479
MF 223828307
J1 148674187
79 31503355"""

l = [x.strip().split() for x in data.split("\n")]
#l = [(y[0], institution_lookup[y[1]]) for y in l]
l = [(y[0], y[1]) for y in l]
counts, institutions = zip(*l)
#sorted_inst = sorted(institutions, key=lambda x: counts[institutions.index(x)])

output_file("plot_dists.html", title="Instituion Dist Plots")

plot_height = 300
plot_width = 650

p = figure(x_range=institutions, height=plot_height, width=plot_width, title="Institution Slide Counts")
p.vbar(x=institutions, top=counts, width=0.9)
p.xgrid.grid_line_color = None
p.y_range.start = 0
#p.xaxis.major_label_orientation = 0
#show(p)

l = [x.strip().split() for x in slide_data.split("\n")]
#l = [(y[0], institution_lookup[y[1]]) for y in l]
l = [(y[0], y[1]) for y in l]
counts, institutions = zip(*l)

p_slide = figure(x_range=institutions, height=plot_height, width=plot_width, title="Institution Tile Counts")
p_slide.vbar(x=institutions, top=counts, width=0.9)
p_slide.xgrid.grid_line_color = None
p_slide.y_range.start = 0
#p_slide.xaxis.major_label_orientation = np.pi / 2.0
#p_slide.xaxis.major_label_orientation = 1.2
#show(p)


l = [x.strip().split() for x in file_data.split("\n")]
#l = [(institution_lookup[y[0]], y[1]) for y in l]
l = [(y[0], y[1]) for y in l]
institutions, counts = zip(*l)
p_disk = figure(x_range=institutions, height=plot_height, width=plot_width, title="Institution Data Amount")
p_disk.vbar(x=institutions, top=counts, width=0.9)
p_disk.xgrid.grid_line_color = None
p_disk.y_range.start = 0
p_disk.xaxis.major_label_orientation = np.pi / 2.0
p_disk.yaxis.formatter = CustomJSTickFormatter(code='''return (tick/1000000000) + ' ' + 'GB';''')
#show(p)


gp = layout([p], [p_slide], [p_disk])
save(gp)
