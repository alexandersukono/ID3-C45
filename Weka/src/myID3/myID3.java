package myID3;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;

public class myID3 extends Classifier {
  
  private myID3[] successor;
  private double classValue;
  private double[] distribution;
  private Attribute attribute;
  private Attribute classAttribute;

  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }
  
  private Instances[] splitInst(Instances data, Attribute att) {

    Instances[] splitInst = new Instances[att.numValues()];
    for (int j = 0; j < att.numValues(); j++) {
      splitInst[j] = new Instances(data, data.numInstances());
    }
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      splitInst[(int) inst.value(att)].add(inst);
    }
    for (int i = 0; i < splitInst.length; i++) {
      splitInst[i].compactify();
    }
    return splitInst;
  }
  
  private double calculateEntropy(Instances data) throws Exception {

    double [] totalClass = new double[data.numClasses()];
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      totalClass[(int) inst.classValue()]++;
    }
    double entropy = 0;
    int i = 0;
    while (i < data.numClasses()){
      if (totalClass[i] > 0){
        entropy -= totalClass[i] * (Math.log(totalClass[i])/Math.log(2));
      }
      i++;
    }
   
    entropy /= (double) data.numInstances();
    return entropy + (Math.log(data.numInstances()) / Math.log(2));
  }

  private double calculateGain(Instances data, Attribute att) 
    throws Exception {

    double gain = calculateEntropy(data);
    Instances[] splitInst = splitInst(data, att);
    int i = 0;
    while (i<att.numValues()){
      if (splitInst[i].numInstances() > 0) {
        gain -= ((double) splitInst[i].numInstances() / (double) data.numInstances()) * calculateEntropy(splitInst[i]);
      }
      i++;
    }
    return gain;
  }

  private void makeTree(Instances data) throws Exception {

    // Check if no instances have reached this node.
    if (data.numInstances() == 0) {
      attribute = null;
      classValue = Instance.missingValue();
      distribution = new double[data.numClasses()];
      return;
    }

    // Calculate attribute with maximum information gain.
    double[] gains = new double[data.numAttributes()];
    Enumeration attEnum = data.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      gains[att.index()] = calculateGain(data, att);
    }
    attribute = data.attribute(Utils.maxIndex(gains));
    
    // Make leaf if information gain is zero. 
    // Otherwise create successors.
    if (Utils.eq(gains[attribute.index()], 0)) {
      attribute = null;
      distribution = new double[data.numClasses()];
      Enumeration instEnum = data.enumerateInstances();
      while (instEnum.hasMoreElements()) {
        Instance inst = (Instance) instEnum.nextElement();
        distribution[(int) inst.classValue()]++;
      }
      Utils.normalize(distribution);
      classValue = Utils.maxIndex(distribution);
      classAttribute = data.classAttribute();
    } else {
      Instances[] splitInst = splitInst(data, attribute);
      successor = new myID3[attribute.numValues()];
      for (int j = 0; j < attribute.numValues(); j++) {
        successor[j] = new myID3();
        successor[j].makeTree(splitInst[j]);
      }
    }
  }

  @Override
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    makeTree(data);
  }

  @Override
  public double classifyInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("myID3: Cannot Handle Missing Value");
    }
    if (attribute == null) {
      return classValue;
    } else {
      return successor[(int) instance.value(attribute)].
        classifyInstance(instance);
    }
  }

  @Override
  public String toString() {

    if ((distribution == null) && (successor == null)) {
      return "myID3: No model built yet.";
    }
    return "myID3\n\n" + toString(0);
  }

  private String toString(int level) {

    StringBuffer text = new StringBuffer();
    
    if (attribute == null) {
      if (Instance.isMissingValue(classValue)) {
        text.append(": null");
      } else {
        text.append(": " + classAttribute.value((int) classValue));
      } 
    } else {
      for (int j = 0; j < attribute.numValues(); j++) {
        text.append("\n");
        for (int i = 0; i < level; i++) {
          text.append("|  ");
        }
        text.append(attribute.name() + " = " + attribute.value(j));
        text.append(successor[j].toString(level + 1));
      }
    }
    return text.toString();
  }
}
