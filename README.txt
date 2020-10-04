The data folder contains:

##raw-sequences
This is a subfolder containing two files, one for the nox and one for the tox sequences. The structure of each file is as follows.

[Germline Type Code] [ID of the sequence] [Phenotype (i.e., tox or nox)]
Sequence aligned in Kabat-Chothia numbering
Germline aligned in Kabat-Chothia numbering
[blank line]
Sequence capturing the differences between the sequence and corresponding germline (X = same amino acid, else specific mutation)
[blank line]
Subsequent sequence [...]

##arffFiles.zip
This archive contains all training and test set (both with and without balancing with SMOTE filter) used in the presented experiments.
Once decompressed, each subfolder is named with the combination of predictor variables used in the containing training and test sets, using the following mapping:
onegram = AMP features
twogramsDimer = DAP features
twogramsMonomer = MAP features

##code
This folder contains an archive with the code needed to replicate our experiments concerning the 10-fold validation of LICTOR.

To run the code:
1. Decompress the file lictor-experiments.zip
2. Make sure that on your machine there is Java 8 and Maven installed
3. cd into the decompressed folder
4. run "maven clean"
5. run "maven package"
6. The compiled code will be in target/lictor-experiments-1.0-jar-with-dependencies.jar
7. cd into target
8. run "java -jar pathArffFiles pathOutputFile.csv", where pathArffFiles is the absolute path of the folder obtained by decompressing arffFiles.zip
and pathOutputFile.csv is the absolute path of a CSV file that will report the achieved results. 

The code has been written using the Eclipse IDE and tested with Java 8. Any Java IDE can be used to inspect its code. We recommend to start reading the code from the file main/Main.java.

##output
The structure of the CSV file reporting the achieved results is as follows.
[Machine learner used in the specific experiment],[Feature list used],[SMOTE filter (i.e., TRUE= used, FALSE= not used)],[each of the 10 fold used as test set],[number of True Positive (TP)],[number of True Negative (TN)],[number of False Positive (FP)],[number of False Negative (FN)],[Sensitivity],[Specificity],[AUC]

##expected runtime
At most three hours. However, even partial results are visible in the output file after a few (10) minutes.