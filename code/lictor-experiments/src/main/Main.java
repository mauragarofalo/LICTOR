package main;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import machineLearning.EvaluationUtilities;
import weka.classifiers.Evaluation;

public class Main {

	public static void main(String[] args) throws Exception {
		
		//dataFolder is the absolute path of the decompressed arffFiles.zip file provided
		//with this code
		File dataFolder = new File(args[0]);
		
		//Categories to predict, true -> tox, false -> nox
		ArrayList<String> categories = new ArrayList<String>();
		categories.add("false");
		categories.add("true");
		
		//Classifiers used in the experiments
		ArrayList<String> classifiersToTest = new ArrayList<String>();
		classifiersToTest.add("RandomForest");
		classifiersToTest.add("Bayesian");
		classifiersToTest.add("J48");
		classifiersToTest.add("Logistic");
		
		//Absolute path of the CSV file where to write the results
		File output = new File(args[1]);
		PrintWriter pw = new PrintWriter(output);
		pw.println("machine_learner,features,smote,feature_selection,fold,TP,TN,FP,FN,sensitivity,specificity,AUC");
		
		File[] arffFiles = dataFolder.listFiles();
		
		boolean FEATURE_SELECTION = true;
		
		for (String classifier : classifiersToTest) {
			System.out.println("Experimenting " + classifier + " classifier");
			for (int smoteIndex = 0; smoteIndex <= 1; smoteIndex++) {
				boolean SMOTE = false;
				if (smoteIndex == 1)
					SMOTE = true;

				System.out.println("SMOTE is " + SMOTE);
				System.out.println("FEATURE_SELECTION is " + FEATURE_SELECTION);

				for (File featuresFolder : arffFiles) {
					if (featuresFolder.isDirectory() && !featuresFolder.getName().startsWith(".")) {
						String features = featuresFolder.getName();
						System.out.println("Features are " + features);

						Map<String, Double> results = new HashMap<String, Double>();
						results.put("evaluated instances", 0.0);
						results.put("true positive", 0.0);
						results.put("true negative", 0.0);
						results.put("false positive", 0.0);
						results.put("false negative", 0.0);
						results.put("AUC", 0.0);

						for (int i = 1; i <= 10; i++) {
							System.out.println("Fold " + i);
							File training = new File(featuresFolder.getAbsolutePath() + "/training_" + i + ".arff");
							if (SMOTE)
								training = new File(
										featuresFolder.getAbsolutePath() + "/training_" + i + "_balanced.arff");
							File test = new File(featuresFolder.getAbsolutePath() + "/test_" + i + ".arff");

							Evaluation evaluation = EvaluationUtilities.getResultsForAclassifier(
									training.getAbsolutePath(), test.getAbsolutePath(), "iswtox", categories,
									classifier, false, FEATURE_SELECTION);
							Map<String, Double> resultsInThisFold = EvaluationUtilities
									.getEvaluationMetrics(evaluation);

							results.put("evaluated instances",
									results.get("evaluated instances") + resultsInThisFold.get("evaluated instances"));
							results.put("true positive",
									results.get("true positive") + resultsInThisFold.get("true positive"));
							results.put("true negative",
									results.get("true negative") + resultsInThisFold.get("true negative"));
							results.put("false positive",
									results.get("false positive") + resultsInThisFold.get("false positive"));
							results.put("false negative",
									results.get("false negative") + resultsInThisFold.get("false negative"));
							results.put("AUC", results.get("AUC") + evaluation.areaUnderROC(1));

							double TP = resultsInThisFold.get("true positive");
							double TN = resultsInThisFold.get("true negative");
							double FP = resultsInThisFold.get("false positive");
							double FN = resultsInThisFold.get("false negative");

							double sensitivity = TP / (TP + FN);
							double specificity = TN / (TN + FN);

							pw.println(classifier + "," + features + "," + SMOTE + "," + FEATURE_SELECTION + "," + i
									+ "," + TP + "," + TN + "," + FP + "," + FN + "," + sensitivity + "," + specificity
									+ "," + evaluation.areaUnderROC(1));
							pw.flush();
						}

						double TP = results.get("true positive");
						double TN = results.get("true negative");
						double FP = results.get("false positive");
						double FN = results.get("false negative");

						double sensitivity = TP / (TP + FN);
						double specificity = TN / (TN + FN);

						pw.println(classifier + "," + features + "," + SMOTE + "," + FEATURE_SELECTION + "," + "overall"
								+ "," + TP + "," + TN + "," + FP + "," + FN + "," + sensitivity + "," + specificity
								+ "," + results.get("AUC") / 10);
						pw.flush();
					}
				}
			}
		}

		pw.close();
	}
	
	
}
