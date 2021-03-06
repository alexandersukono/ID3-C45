/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.*;
import myWeka.myID3;
import myWeka.myJ48;

/**
 *
 * @author Alex
 */
public class Weka {

    
    public static Instances loadFile(String name) throws Exception {
        DataSource source = new DataSource("data/" + name);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        System.out.println(data.toString());
        return data;      
    }
    
    public static void removeAttr(Instances data, int id) {
        data.deleteAttributeAt(id);
    }
    
    public static Instances resample(Instances data) {
	final Resample filter = new Resample();
	Instances filteredIns = null;
	filter.setBiasToUniformClass(5.0);
	try {
		filter.setInputFormat(data);
		filter.setNoReplacement(false);
		filter.setSampleSizePercent(100);
		filteredIns = Filter.useFilter(data, filter);
	} catch (Exception e) {
		System.out.println("Error when resampling input data!");
	}
	return filteredIns;
    }
    
    public static void buildClassifier(Classifier cModel, Instances data) throws Exception {
        cModel.buildClassifier(data);
    }
    
    public static void evaluateModel(Classifier cModel,Instances trainingData, Instances testData) throws Exception {
        Evaluation eTest = new Evaluation(trainingData);
        eTest.evaluateModel(cModel, testData);
        String strSummary = eTest.toSummaryString();
        String strAccuracy = eTest.toClassDetailsString();
        String strConfusion = eTest.toMatrixString();
        System.out.println();
        System.out.println("==== Results =====");
        System.out.println(strSummary);
        System.out.println(strAccuracy);
        System.out.println(strConfusion);
    }
    
    public static Instances[] percentSplit(Instances data, double percentage) {
        data.randomize(new java.util.Random(0));
        int trainSize = (int) Math.round(data.numInstances() * percentage);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        Instances[] instances = new Instances[2];
        instances[0] = train;
        instances[1] = test;
        System.out.println("==== Training Data ====");
        System.out.println(train.toString());
        System.out.println();
        System.out.println("==== Testing Data ====");
        System.out.println(test.toString());
        return instances;
    }
    
    public static void crossValidation(Classifier cModel, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(cModel, data, 10, new Random(1));
        String strSummary = eval.toSummaryString();
        String strAccuracy = eval.toClassDetailsString();
        String strConfusion = eval.toMatrixString();
        System.out.println();
        System.out.println("==== Results =====");
        System.out.println(strSummary);
        System.out.println(strAccuracy);
        System.out.println(strConfusion);
    }
    
    private static void saveModel(Classifier c, String name) throws Exception {
        ObjectOutputStream oos;
        oos = new ObjectOutputStream(new FileOutputStream("weka_models/" + name + ".model"));
        oos.writeObject(c);
        oos.flush();
        oos.close();
        System.out.println("Model has been saved");
    }
    
     private static Classifier loadModel(String name) throws Exception {
        Classifier classifier;

        FileInputStream fis = new FileInputStream("weka_models/" + name + ".model");
        ObjectInputStream ois = new ObjectInputStream(fis);

        classifier = (Classifier) ois.readObject();
        ois.close();
        System.out.println("Model has been loaded");
        return classifier;
    }
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        Classifier cModel = null;
        Instances data;
        Instances testData;
        String filename;
        double splitPercentage;
        
        System.out.print("Input data file name : ");
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
        filename = input.readLine();
        data = loadFile(filename);
        
        while (true) {
            System.out.println();
            System.out.println("==== Select Option ====");
            System.out.println("1. Remove Attribute");
            System.out.println("2. Filter : Resample");
            System.out.println("3. Build Id3 Classifier");
            System.out.println("4. Build J48 Classifier");
            System.out.println("5. Testing Model Given Data Set");
            System.out.println("6. 10-fold Cross Validation");
            System.out.println("7. Percentage Split");
            System.out.println("8. Save Model");
            System.out.println("9. Load Model");
            System.out.println("10. Classify Input Data");
            System.out.println("11. Build myID3 Classifier");
            System.out.println("12. Build myJ48 Classifier");
            System.out.println("*** Remember to build classifier first before evaluating or testing data set");

            System.out.print("Your choice : ");
            int option = Integer.parseInt(input.readLine());
            System.out.println();
            switch(option) {
            case 1 :
                System.out.print("Input attribute number to remove : ");
                int attribute = Integer.parseInt(input.readLine());
                removeAttr(data, attribute);
                System.out.println(data.toString());
                break;
            case 2 :
                data = resample(data);
                System.out.println("After resample");
                System.out.println(data.toString());
                break;
            case 3 :
                cModel = (Classifier)new Id3();
                buildClassifier(cModel, data);
                System.out.println("Id3 classifier has been built");
                break;
            case 4 : 
                cModel = (Classifier)new J48();
                buildClassifier(cModel, data);
                System.out.println("J48 classifier has been built");
                break;
            case 5 :
                System.out.print("Input test data file name : ");
                filename = input.readLine();
                testData = loadFile(filename);
                evaluateModel(cModel,data,testData);
                break;
            case 6 :
                crossValidation(cModel,data);
                break;
            case 7 :
                System.out.print("Input split percentage : ");
                splitPercentage = Double.parseDouble(input.readLine())/100;
                Instances[] instances = percentSplit(data, splitPercentage);
                System.out.print("Input test data file name : ");
                evaluateModel(cModel,instances[0],instances[1]);
                break;
            case 8 :
                System.out.print("Input model name to save : ");
                filename = input.readLine();
                saveModel(cModel, filename);
                break;
            case 9 :
                System.out.print("Input model name to load : ");
                filename = input.readLine();
                cModel = loadModel(filename);
                break;
            case 10 : 
                Instances unlabeledData = loadFile("test.arff");
 
                System.out.println("Classifier Result : ");
                for (int i = 0; i < unlabeledData.numInstances(); i++) {
                    double clsLabel = cModel.classifyInstance(unlabeledData.instance(i));
                    System.out.println(unlabeledData.classAttribute().value((int) clsLabel));
                }
                break;
            case 11 : 
                cModel = (Classifier)new myID3();
                buildClassifier(cModel, data);
                System.out.println("myID3 classifier has been built");
                break;
            case 12 : 
                cModel = (Classifier)new myJ48();
                buildClassifier(cModel, data);
                System.out.println("myJ48 classifier has been built");
                break;
            default :
                System.out.println("Invalid Option");
                break;
            }
        }
    }
}
