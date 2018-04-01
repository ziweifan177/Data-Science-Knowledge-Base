/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package OperationPackage;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.TreeMap;

/**
 *
 * @author Wing Fan
 */
public class FileLoader {

    /**
     * Get the total number of words count in one text file.
     */
    public static int getWordNum(String inputfile) throws FileNotFoundException {

        TreeMap<Integer, String> map = FileLoader.wordTreemapGenerate(inputfile);
         return map.size();
    }

    /**
     * Generate a Treemap for word.
     */
    public static TreeMap wordTreemapGenerate(String inputfile) throws FileNotFoundException {

        TreeMap<Integer, String> map = new TreeMap<Integer, String>();
        Scanner scanner = new Scanner(new File(inputfile));
        int key = 0;

        while (scanner.hasNext()) {
            String word = scanner.next().replaceAll("[^a-zA-Z]+", "");
            key++;
            map.put(key, word);
        }
        scanner.close();
        return map;
    }

    /**
     * Generate a Treemap for punctuation.
     */
    public static TreeMap punctuationTreemapGenerate(String inputfile) throws FileNotFoundException {

        TreeMap<Integer, String> map = new TreeMap<Integer, String>();
        Scanner scanner = new Scanner(new File(inputfile));
        int key = 0;

        while (scanner.hasNext()) {
            String punctuation = scanner.next().replaceAll("[A-Za-z0-9]", "");
            key++;
            map.put(key, punctuation);
        }
        scanner.close();
        return map;
    }
}
