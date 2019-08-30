/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package javaappweka;

import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author G4_Homes
 */
public class nbayes {

    public static void main(String args[]) throws IOException, Exception {
        DataSource file = new DataSource("C:\\Users\\G4_Homes\\Music\\interest.arff");
        Instances dataset = file.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        NaiveBayes nby = new NaiveBayes();
        nby.buildClassifier(dataset);
        
        Evaluation eva = new Evaluation(dataset);
        eva.evaluateModel(nby, dataset, (Object[]) args);
        
        System.out.println(eva.toSummaryString());
        System.out.println(eva.toMatrixString());
    }

}
