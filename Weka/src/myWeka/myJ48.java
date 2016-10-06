package myWeka;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;
import weka.core.AttributeStats;

/**
 *
 * @author tegar
 */
public class myJ48 extends Classifier {

    Instances instances;
    static final long serialVersionUID = -2693678647096322561L;
    private myJ48[] child;
    private Attribute attrSplit;
    private Attribute attrClass;
    public int leafIdx;
    private double[] leafDistribution;

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        
        Instances noMissingValueData = handleMissingValue(data);

        for (int i = 0; i < noMissingValueData.numInstances(); i++) {
            System.out.println(noMissingValueData.instance(i).toString());
        }
        makeTree(noMissingValueData);
    }
    
    private double getOptimumThreshold(Instances data, Attribute attribute) throws Exception {
        double[] threshold = new double[data.numInstances()];
        double[] gainRatio = new double[data.numInstances()];
        for (int i = 0; i < data.numInstances() - 1; ++i) {
            if (data.instance(i).classValue() != data.instance(i + 1).classValue()) {
                threshold[i] = (data.instance(i).value(attribute) + data.instance(i + 1).value(attribute)) / 2;
                gainRatio[i] = computeGainRatio(data, attribute, threshold[i]);
            }
        }
        double result = (double) threshold[Utils.maxIndex(gainRatio)];
        return result;
    }
    
    public int maxAttr(Instances data, Attribute attr) throws Exception {
        int[] maxval = new int[attr.numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            Instance temp = data.instance(i);
            maxval[(int) temp.classValue()]++;
        }
        return Utils.maxIndex(maxval);
    }

    private Instances handleMissingValue(Instances oldData) {
        Instances data = oldData;
        Enumeration enumAttr = data.enumerateAttributes();
        while (enumAttr.hasMoreElements()) {
            Attribute attr = (Attribute) enumAttr.nextElement();
            
            //Kasus data nominal dengan melakukan assignment nilai mayoritas
            if (attr.isNominal()) {
                AttributeStats attributeStats = data.attributeStats(attr.index());
                int maxIndex = 0;
                
                for (int i = 1; i < attr.numValues(); i++) {
                    if (attributeStats.nominalCounts[maxIndex] < attributeStats.nominalCounts[i]) {
                        maxIndex = i;
                    }
                }
                
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance instance = (Instance) data.instance(i);

                    if (instance.isMissing(attr.index())) {
                        instance.setValue(attr.index(), maxIndex);
                    }
                }
            } 
            //Kasus data numerik dengan melakukan assignment mean dari instansi atribut
            else if (attr.isNumeric()) {
                AttributeStats attributeStats = data.attributeStats(attr.index());
                double mean = attributeStats.numericStats.mean;
                if (Double.isNaN(mean)) {
                    mean = 0;
                }
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance instance = (Instance) data.instance(i);
                    if (instance.isMissing(attr.index())) {
                        instance.setValue(attr.index(), (int) mean);
                    }
                }
            }
        }
        return data;
    }

    private Instance handleMissingValue(Instance oldData) {
        Instance instance = oldData;
        Instances data = instances;
        Enumeration enumAttr = data.enumerateAttributes();
        while (enumAttr.hasMoreElements()) {
            Attribute attr = (Attribute) enumAttr.nextElement();
            
            //Kasus data nominal dengan melakukan assignment nilai mayoritas
            if (attr.isNominal()) {
                AttributeStats attributeStats = data.attributeStats(attr.index());
                int maxIndex = 0;
                for (int i = 1; i < attr.numValues(); i++) {
                    if (attributeStats.nominalCounts[maxIndex] < attributeStats.nominalCounts[i]) {
                        maxIndex = i;
                    }
                }
                if (instance.isMissing(attr.index())) {
                    instance.setValue(attr.index(), maxIndex);
                }
            } 
            //Kasus data numerik dengan melakukan assignment mean dari instansi atribut
            else if (attr.isNumeric()) {
                AttributeStats attributeStats = data.attributeStats(attr.index());
                double mean = attributeStats.numericStats.mean;
                if (Double.isNaN(mean)) {
                    mean = 0;
                }
                if (instance.isMissing(attr.index())) {
                    instance.setValue(attr.index(), (int) mean);
                }
            }
        }
        return instance;
    }
    
     public double computeGainRatio(Instances data, Attribute attr) throws Exception {
        double infoGain = 0.0;
        Instances[] splitData = myJ48.this.splitData(data, attr);
        infoGain = computeEntropy(data);
        for (int i = 0; i < attr.numValues(); i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain -= (double) splitData[i].numInstances() / (double) data.numInstances() * computeEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

    public double computeGainRatio(Instances data, Attribute attr, double threshold) throws Exception {
        double infoGain = 0.0;
        Instances[] splitData = splitData(data, attr, threshold);
        infoGain = computeEntropy(data);
        for (int i = 0; i < 2; i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain = infoGain - (double) splitData[i].numInstances() / (double) data.numInstances() * computeEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

    public Instances[] splitData(Instances data, Attribute attr, double threshold) throws Exception {
        Instances[] splitedData = new Instances[2];
        for (int i = 0; i < 2; i++) {
            splitedData[i] = new Instances(data, data.numInstances());
        }

        Enumeration Iterator = data.enumerateInstances();
        while (Iterator.hasMoreElements()) {
            Instance instance = (Instance) Iterator.nextElement();
            if (instance.value(attr) >= threshold) {
                splitedData[1].add(instance);
            } else {
                splitedData[0].add(instance);
            }
        }

        for (Instances instances : splitedData) {
            instances.compactify();
        }

        return splitedData;
    }

    public Instances[] splitData(Instances data, Attribute attr) {
        Instances[] splitedData = new Instances[attr.numValues()];
        for (int i = 0; i < attr.numValues(); i++) {
            splitedData[i] = new Instances(data, data.numInstances());
        }

        Enumeration Iterator = data.enumerateInstances();
        while (Iterator.hasMoreElements()) {
            Instance instance = (Instance) Iterator.nextElement();
            splitedData[(int) instance.value(attr)].add(instance);
        }

        for (Instances instances : splitedData) {
            instances.compactify();
        }

        return splitedData;
    }

    public double computeEntropy(Instances data) {
        if (data.numInstances() == 0) {
            return 0.0;
        }

        double[] classCounts = new double[data.numClasses()];
        Enumeration Iterator = data.enumerateInstances();
        int totalInstance = 0;
        while (Iterator.hasMoreElements()) {
            Instance inst = (Instance) Iterator.nextElement();
            classCounts[(int) inst.classValue()]++;
            totalInstance++;
        }
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            double fraction = classCounts[j] / totalInstance;
            if (fraction != 0) {
                entropy -= fraction * Utils.log2(fraction);
            }
        }

        return entropy;
    }

    public double computeError(Instances instances) throws Exception {
        int correctInstances = 0;
        int incorrectInstances = 0;
        Enumeration enumeration = instances.enumerateInstances();
        while (enumeration.hasMoreElements()) {
            Instance instance = (Instance) enumeration.nextElement();
            if (instance.classValue() == classifyInstance(instance)) {
                correctInstances++;
            } else {
                incorrectInstances++;
            }
        }
        return (double) incorrectInstances / (double) (incorrectInstances + correctInstances);
    }

    private void makeTree(Instances data) throws Exception {
        instances = data;
        if (data.numInstances() == 0) {
            leafIdx = -1;
            leafDistribution = new double[data.numClasses()];
            attrSplit = null;
            return;
        }

        // Menghitung nilai maksimum gain ratio
        double[] gainRatio = new double[data.numAttributes()];
        Enumeration enumAttr = data.enumerateAttributes();
        while (enumAttr.hasMoreElements()) {
            Attribute attr = (Attribute) enumAttr.nextElement();
            if (attr.isNominal()) {
                gainRatio[attr.index()] = computeGainRatio(data, attr);
            } else if (attr.isNumeric()) {
                gainRatio[attr.index()] = computeGainRatio(data, attr, getOptimumThreshold(data, attr));
            }
        }
        
        // Membuat leaf ketika gain ratio itu 0, selain itu membuat suksesor
        if (Utils.eq(gainRatio[Utils.maxIndex(gainRatio)], 0)) {
            attrSplit = null;
            leafDistribution = new double[data.numClasses()];
            Enumeration enumInstances = data.enumerateInstances();
            while (enumInstances.hasMoreElements()) {
                Instance inst = (Instance) enumInstances.nextElement();
                leafDistribution[(int) inst.classValue()]++;
            }
            Utils.normalize(leafDistribution);
            leafIdx = Utils.maxIndex(leafDistribution);
            attrClass = data.classAttribute();
        } else {
            attrSplit = data.attribute(Utils.maxIndex(gainRatio));
            Instances[] splitData;
            int numChild;
            if (attrSplit.isNominal()) {
                numChild = attrSplit.numValues();
                splitData = splitData(data, attrSplit);
            } else {
                numChild = 2;
                splitData = splitData(data, attrSplit, getOptimumThreshold(data, attrSplit));
            }
            
            child = new myJ48[numChild];
            for (int j = 0; j < numChild; j++) {
                child[j] = new myJ48();
                child[j].makeTree(splitData[j]);
                if (Utils.eq(splitData[j].numInstances(), 0)) {
                    child[j].leafIdx = maxAttr(data, data.classAttribute());
                }
            }

            for (int i = 0; i < numChild; i++) {
                if (child[i].leafIdx != 0 && Utils.eq(child[i].leafIdx, -999)) {
                    double[] classDistribution = new double[data.numClasses()];
                    Enumeration instanceEnum = data.enumerateInstances();
                    while (instanceEnum.hasMoreElements()) {
                        Instance instance = (Instance) instanceEnum.nextElement();
                        classDistribution[(int) instance.classValue()]++;
                    }
                    Utils.normalize(classDistribution);
                    child[i].leafIdx = Utils.maxIndex(classDistribution);
                    child[i].leafDistribution = classDistribution;
                }
            }
            pruneTree();
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (attrSplit == null) {
            {
                if (!Utils.eq(leafIdx, Double.NaN)) {
                    return leafIdx;
                } else {
                    Enumeration enumerate = instance.enumerateAttributes();
                    return instance.value(attrClass);
                }
            }
        } else {
            if (attrSplit.isNumeric()) {
                int numericAttrIdx = -1;
                if (instance.value(attrSplit) > getOptimumThreshold(instances, attrSplit)) {
                    numericAttrIdx = 1;
                } else {
                    numericAttrIdx = 0;
                }
                return child[(int) numericAttrIdx].classifyInstance(instance);
            } else if (attrSplit.isNominal()) {
                return child[(int) instance.value(attrSplit)].classifyInstance(instance);
            } else {
                throw new Exception("Exception");
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (attrSplit != null) {
            double splitAttributeIdx = 0;
            if (attrSplit.isNominal()) {
                splitAttributeIdx = instance.value(attrSplit);
                if (Double.isNaN(splitAttributeIdx)) {
                    Instances[] instancesSplitted = splitData(instances, attrSplit);
                    int largestNumIdx = -1;
                    int cnt = 0;
                    for (int i = 0; i < instancesSplitted.length; ++i) {
                        int tmp = instancesSplitted[i].numInstances();
                        if (tmp > cnt) {
                            largestNumIdx = i;
                        }
                    }
                    splitAttributeIdx = largestNumIdx;
                }
                if (splitAttributeIdx == -1) {
                    throw new Exception("This will never happens, sure");
                }
            } else if (attrSplit.isNumeric()) {
                double val = instance.value(attrSplit);
                if (Double.isNaN(val)) {
                    instance = handleMissingValue(instance);
                    val = instance.value(attrSplit);
                }
           
                if (val >= getOptimumThreshold(instances, attrSplit)) {
                    splitAttributeIdx = 1;
                } else {
                    splitAttributeIdx = 0;
                }
            }
            if (child.length > 0) {
                return child[(int) splitAttributeIdx].distributionForInstance(instance);
            }
            if (leafDistribution != null) {
                return leafDistribution;
            } 
        } else {
            return leafDistribution;
        }
        if (leafDistribution != null) {
            return leafDistribution;
        } else {
            return null;
        }
    }

   
    private void pruneTree() throws Exception {
        if (child != null) {
            double beforePruningError = this.computeError(instances);

            double[] classDistribution = new double[instances.numClasses()];
            Enumeration EnumInstance = instances.enumerateInstances();
            while (EnumInstance.hasMoreElements()) {
                Instance instance = (Instance) EnumInstance.nextElement();
                classDistribution[(int) instance.classValue()]++;
            }
            Utils.normalize(classDistribution);
            int idxClass = Utils.maxIndex(classDistribution);

            int correctInstances = 0;
            int incorrectInstances = 0;
            Enumeration enumeration = instances.enumerateInstances();
            while (enumeration.hasMoreElements()) {
                Instance instance = (Instance) enumeration.nextElement();
                if (instance.classValue() == classifyInstance(instance)) {
                    correctInstances++;
                } else {
                    incorrectInstances++;
                }
            }
            double afterPruningError = (double) incorrectInstances / (double) (correctInstances + incorrectInstances);
            if (beforePruningError > afterPruningError) {
                System.out.println("Pruning");
                child = null;
                attrSplit = null;
                leafIdx = idxClass;
                leafDistribution = classDistribution;
            }
        }
    }
}
