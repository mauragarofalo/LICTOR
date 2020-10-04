package machineLearning;

import java.io.FileReader;

import machineLearning.exceptions.InvalidClassifierException;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class BuildClassifiers {

	public static RandomForest buildRandomForestClassifier(String trainingFilePath, String toPredictName) throws Exception{
		
		FileReader frTraining = new FileReader(trainingFilePath);
		Instances ts = new Instances(frTraining);
		
		RandomForest classifier = new RandomForest();
		String options = ("-I 100 -K 0 -S 1");
		String[] optionsArray = options.split(" ");
		classifier.setOptions(optionsArray);
		ts.setClass(ts.attribute(toPredictName));
		classifier.buildClassifier(ts);
		return classifier;
	}
	
	public static Logistic buildLogistic(String trainingFilePath, String toPredictName) throws Exception{
		
		FileReader frTraining = new FileReader(trainingFilePath);
		Instances ts = new Instances(frTraining);
		
		Logistic classifier = new Logistic();
	    
		ts.setClass(ts.attribute(toPredictName));
		classifier.buildClassifier(ts);
		
		return classifier;
	}
	
	public static BayesNet buildBayesianClassifier(String trainingFilePath, String toPredictName) throws Exception{
		
		FileReader frTraining = new FileReader(trainingFilePath);
		Instances ts = new Instances(frTraining);
		
		BayesNet classifier = new BayesNet();
	    
		ts.setClass(ts.attribute(toPredictName));
		classifier.buildClassifier(ts);
		
		return classifier;
	}
	
	public static J48 buildJ48Classifier(String trainingFilePath, String toPredictName) throws Exception{
		
		FileReader frTraining = new FileReader(trainingFilePath);
		Instances ts = new Instances(frTraining);
		
		J48 classifier = new J48();
	    String options = ("-C 0.25 -M 2");
		String[] optionsArray = options.split(" ");
		classifier.setOptions(optionsArray);
		
		ts.setClass(ts.attribute(toPredictName));
		classifier.buildClassifier(ts);
		
		return classifier;
	}
	
	
	public static AttributeSelectedClassifier buildClassifierWithFeatureSelection(String trainingFilePath, String toPredictName, String classifierName) throws Exception{
		
		FileReader frTraining = new FileReader(trainingFilePath);
		Instances ts = new Instances(frTraining);
		ts.setClass(ts.attribute(toPredictName));
		
		AttributeSelectedClassifier selectionClassifier = new AttributeSelectedClassifier();
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setThreshold(0.01);
		
		AbstractClassifier classifier = null; 
		switch(classifierName){
			case "RandomForest": classifier = buildRandomForestClassifier(trainingFilePath, toPredictName); break;
			case "Bayesian": classifier = buildBayesianClassifier(trainingFilePath, toPredictName); break;
			case "J48": classifier = buildJ48Classifier(trainingFilePath, toPredictName); break;
			case "Logistic": classifier = buildLogistic(trainingFilePath, toPredictName); break;
		default: throw new InvalidClassifierException("The classifier " + classifierName + " cannot be built");
		}
		
		selectionClassifier.setClassifier(classifier);
		selectionClassifier.setEvaluator(eval);
		selectionClassifier.setSearch(ranker);
		

		selectionClassifier.buildClassifier(ts);
		
		return selectionClassifier;
	}
	
}
