/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
/**
 *
 * @author Alex
 */
public class Weka {

    
    public static void loadFile() {
        try {
            DataSource source = new DataSource("/some/where/data.arff");
            Instances data = source.getDataSet();
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
        } catch(Exception e) {
            System.out.println("Exception thrown : " + e);
        }
        
        
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        
    }
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        loadFile();
    }
}
