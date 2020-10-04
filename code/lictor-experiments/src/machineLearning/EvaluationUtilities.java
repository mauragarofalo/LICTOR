package machineLearning;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import machineLearning.exceptions.InvalidClassifierException;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.instance.RemoveDuplicates;

public class EvaluationUtilities {

	public static Evaluation getResultsForAclassifier(String trainingFilePath, String testFilePath, String toPredictName, 
			ArrayList<String> categories, String classifierName, 
			boolean byBalancingTrainingSet, boolean withFeatureSelection) throws Exception{
		
		if(byBalancingTrainingSet){
			trainingFilePath = balanceTrainingSetWithSMOTE(trainingFilePath, toPredictName, categories);
			if(trainingFilePath == null)
				return null;
		}
		
		AbstractClassifier classifier = null; 
		switch(classifierName){
		case "RandomForest": 
			if(withFeatureSelection)
				classifier = BuildClassifiers.buildClassifierWithFeatureSelection(trainingFilePath, toPredictName, classifierName);
			else
				classifier = BuildClassifiers.buildRandomForestClassifier(trainingFilePath, toPredictName); 
			break;
		case "Bayesian": 
			if(withFeatureSelection)
				classifier = BuildClassifiers.buildClassifierWithFeatureSelection(trainingFilePath, toPredictName, classifierName);
			else
				classifier = BuildClassifiers.buildBayesianClassifier(trainingFilePath, toPredictName); 
			break;
		case "J48": 
			if(withFeatureSelection)
				classifier = BuildClassifiers.buildClassifierWithFeatureSelection(trainingFilePath, toPredictName, classifierName);
			else
				classifier = BuildClassifiers.buildJ48Classifier(trainingFilePath, toPredictName); 
			break;
		case "Logistic": 
			if(withFeatureSelection)
				classifier = BuildClassifiers.buildClassifierWithFeatureSelection(trainingFilePath, toPredictName, classifierName);
			else
				classifier = BuildClassifiers.buildLogistic(trainingFilePath, toPredictName); 
			break;
		default: throw new InvalidClassifierException("The classifier " + classifierName + " cannot be built");
		}
		
		Evaluation toReturn = runOnTestSet(classifier, testFilePath, toPredictName);
		System.out.println("[COMPLETED] " + trainingFilePath + ": " + classifierName + " - feature selection=" + withFeatureSelection + " - balancing training=" + byBalancingTrainingSet);
		
		return toReturn;
	}
	
	
	public static Evaluation runOnTestSet(Classifier classifier, String testFilePath, String toPredict) throws Exception{
		
		FileReader frTest = new FileReader(testFilePath);
		Instances test = new Instances(frTest);
		test.setClass(test.attribute(toPredict));
		
		StringBuffer predsBuffer = new StringBuffer();
		PlainText pt = new PlainText();
		pt.setBuffer(predsBuffer);
		pt.setHeader(test);
		Evaluation eval = new Evaluation(test);
		eval.evaluateModel(classifier, test, pt);
		
		return eval;
	}
	
	public static Map<String, Double> getEvaluationMetrics(Evaluation results){
		Map<String, Double> metrics = new HashMap<String, Double>();
		
		double confusionMatrix[][] = results.confusionMatrix();
		
		double tn = confusionMatrix[0][0];
		double tp = confusionMatrix[1][1];
		double fp = confusionMatrix[0][1];
		double fn = confusionMatrix[1][0];
		double total = tp + tn + fp + fn; 
		
		metrics.put("evaluated instances", total);
		metrics.put("true positive", tp);
		metrics.put("true negative", tn);
		metrics.put("false positive", fp);
		metrics.put("false negative", fn);
		
		return metrics;
	}
	
	public static String balanceTrainingSetWithSMOTE(String trainingFilePath, String toPredictName, ArrayList<String> categories) throws Exception{
		
		FileReader frTraining = new FileReader(trainingFilePath);
		Instances instances = new Instances(frTraining);
		instances.setClass(instances.attribute(toPredictName));
		
		ArrayList<Integer> percentages = getPercentageOfArtificialInstancesNeeded(instances, categories);
		if(percentages == null)
			return null;
			
		for(int j=0; j<percentages.size(); j++){
			if(percentages.get(j)==0)
				continue;
				
			SMOTE filter =new SMOTE();
			filter.setInputFormat(instances);
			String options = ("-C " + (j+1) + " -K 5 -P " + percentages.get(j) + " -S 1");
			String[] optionsArray = options.split(" ");
			filter.setOptions(optionsArray);
			instances = Filter.useFilter(instances, filter);
				
		}
			
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		File outputFile = new File(trainingFilePath.replace(".arff", "_balanced.arff"));
		saver.setFile(outputFile);
		saver.writeBatch();

		return outputFile.getAbsolutePath();
		
	}
	
	private static ArrayList<Integer> getPercentageOfArtificialInstancesNeeded(Instances instances, ArrayList<String> categories){
		ArrayList<Integer> result = new ArrayList<Integer>();
		ArrayList<Integer> numberOfInstancesPerCategory = new ArrayList<Integer>();
		for(int i=0; i<categories.size(); i++){
			numberOfInstancesPerCategory.add(0);
		}
		
		Iterator<Instance> iterator = instances.iterator();
		while(iterator.hasNext()){
			Instance instance = iterator.next();
			int index = ((Double) instance.classValue()).intValue();
			numberOfInstancesPerCategory.set(index,numberOfInstancesPerCategory.get(index)+1);
		}
		
		int max = 0;
		int maxIndex = -1;
		
		for(int i=0; i<numberOfInstancesPerCategory.size(); i++){
			if(numberOfInstancesPerCategory.get(i) < 6)
				return null;
			if(numberOfInstancesPerCategory.get(i) > max){
				max = numberOfInstancesPerCategory.get(i);
				maxIndex = i;
			}
		}
		
		for(int i=0; i<numberOfInstancesPerCategory.size(); i++){
			if(i==maxIndex){
				result.add(0);
			} else {
				int percentage = ((max-numberOfInstancesPerCategory.get(i))*100)/numberOfInstancesPerCategory.get(i);
				result.add(percentage);
			}
		}
		
		return result;
	}
	
	
	public static String createTrainingAndTestSets(int nFolds, String arffFilePath) throws Exception{
		
		File outputDir = new File(arffFilePath.replace(".arff", ""));
		outputDir.mkdir();
	
		BufferedReader br = new BufferedReader(new FileReader(arffFilePath));
		String line = null;
		String header = "";
		
		ArrayList<String> elements = new ArrayList<String>();
		
		boolean isData = false;
		while ((line = br.readLine()) != null) {
			if(isData){
				elements.add(line);
			} else if(line.startsWith("@data")){
				isData = true;
				header+=line + "\n";
			} else {
				header+=line + "\n";
			}
		}
		br.close();
		Collections.shuffle(elements);
		
		int foldSize = elements.size()/nFolds;
		
		int count =0;
		
		for(int i=0; i<elements.size(); i+=foldSize){
			count++;
			File test = new File(outputDir.getAbsolutePath() + "/test_" + count + ".arff");
			PrintWriter pw = new PrintWriter(test);
			pw.println(header);
			
			if(count == nFolds){
				for(int j=i; j<elements.size(); j++){
					pw.println(elements.get(j).replaceAll(" ", "_"));
				}
				pw.close();
			} else {
				for(int j=i; j<i+foldSize; j++){
					if(j<elements.size())
						pw.println(elements.get(j).replaceAll(" ", "_"));
				}
				pw.close();
			}
			
			
			File training = new File(outputDir.getAbsolutePath() + "/training_" + count + ".arff");
			pw = new PrintWriter(training);
			pw.println(header);
			
			
			if(count == nFolds){
				for(int j=0; j<elements.size(); j++){
					if(j<i){
						pw.println(elements.get(j).replaceAll(" ", "_"));
					}
				}
			} else {
				for(int j=0; j<elements.size(); j++){
					if(!(j>=i && j<i+foldSize)){
						pw.println(elements.get(j).replaceAll(" ", "_"));
					}
				}
			}
			pw.close();
			
			if(count == nFolds)
				break;
				
			
		}
		
		return outputDir.getAbsolutePath();
	}
	
	public static Instances removeDuplicatedInstances(Instances startingInstances) throws Exception{
		RemoveDuplicates filter =new RemoveDuplicates();
		filter.setInputFormat(startingInstances);
		return(Filter.useFilter(startingInstances, filter));
	}
	
}
