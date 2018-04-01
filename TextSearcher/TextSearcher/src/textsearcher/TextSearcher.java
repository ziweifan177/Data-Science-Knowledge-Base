/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package textsearcher;

import LogicPackage.SearchEngine;
import static LogicPackage.SearchEngine.search;
import OperationPackage.FileLoader;
import OperationPackage.IndexMatcher;
import OperationPackage.TreemapIterator;

import java.util.TreeMap;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

public class TextSearcher {

    public static void main(String[] args) throws FileNotFoundException {

        System.out.println("Enter search term: ");

        //Initialization: Check the parameter:
        if (args.length < 3 || Pattern.matches("[^\\d]", args[2].trim()) || Integer.parseInt(args[2].trim()) < 0) {
            //EXIT: If no args, or number of surrouding context num is not a integer, or the args is negative.
            System.out.println("Warning: Invalid Args! Exit!");
            System.exit(0);
        }

        //Initialization: Get the number of words in txt.
        String inputfile = args[0].trim();
        Path path = Paths.get(inputfile);

        if (Files.notExists(path)) {
            System.out.println("Warning: The file does not exist! Exit!");
            System.exit(0);
        }

        String wordTarget = args[1].trim();
        System.out.println(wordTarget);

        int wordsNum = FileLoader.getWordNum(inputfile);
        System.out.println("Enter number of surrounding Context Words (Less than the number of " + wordsNum + "): ");
        int numberContextWords = Integer.parseInt(args[2].trim());

        if (numberContextWords > wordsNum) {
            System.out.println("Warning: Invalid number. Exit!");
            System.exit(0); //EXIT: if the args is more than number of the words in txt. 
        }

        //Step 1: Word processing-Retrieve all elements of words:
        System.out.println("Step 1: Word processing-Retrieve all elements of words:\n");

        TreeMap map = FileLoader.wordTreemapGenerate(inputfile);
        TreeMap punctuationMap = FileLoader.punctuationTreemapGenerate(inputfile);

        Set set = map.entrySet();

        ///Get all elements of Words to print:
//        Iterator iterator = set.iterator();
//        while (iterator.hasNext()) {
//            Map.Entry me = (Map.Entry) iterator.next();
//            //System.out.println("Key: " + me.getKey() + " Value: " + me.getValue());
//        }

        //Step 2: Search and locate the word:
        System.out.println("Step 2: Search and locate the word:\n");
        if (TreemapIterator.containsWord(map, wordTarget)) {
            ArrayList indexList = SearchEngine.search(map, wordTarget); //Get all locations of this term, single or multiple.
            System.out.println("The word '" + wordTarget + "' is on the locations of: " + indexList + "\n");

            for (int indexCount = 0; indexCount < indexList.size(); indexCount++) {
                int indexNumber = Integer.parseInt(indexList.get(indexCount).toString());

                if (indexNumber - numberContextWords < 1 || indexNumber + numberContextWords > wordsNum) {
                    // EXIT: It has been out of the range.
                    System.out.println("Warning:No context words according to the number of context words. Exit!");
                    System.exit(0);
                }
            }

            // Step 3: Get all punctuations.
            System.out.println("Step 3: Get all punctuation marks:\n");
         
            ///Get all punctuation marks for printing.
//            Set punctuationSet = punctuationMap.entrySet();
//            Iterator punctuationIterator = punctuationSet.iterator();
//            while (punctuationIterator.hasNext()) {
//                Map.Entry me = (Map.Entry) punctuationIterator.next();
//                //System.out.println("Key: " + me.getKey() + " Value: " + me.getValue());
//            }

            //"Step 4: Get pure String list for located word based on context number.
            System.out.println("Step 4: Get String list for located words based on context number: \n");
            for (int index = 0; index < indexList.size(); index++) {
                String words = null;
                StringBuilder sb = new StringBuilder();
                int indexNum = Integer.parseInt(indexList.get(index).toString());

                for (int countWord = indexNum - numberContextWords; countWord <= indexNum + numberContextWords; countWord++) {
                    if (indexNum - numberContextWords < 1 || indexNum + numberContextWords > wordsNum) {
                        System.out.println("Warning: Another one will not be caculated because it is out of range.");
                        System.exit(0);
                    }
                    
                    String word = map.get(countWord).toString();
                    words = sb.append(word).append(" ").toString().trim();

                    if (countWord == indexNum - numberContextWords) { //For first word: should be with punctuation mark after it.
                        boolean ifFirstPunctuationExist = IndexMatcher.checkFirstPunctuationExist(punctuationMap, countWord);
                        if (ifFirstPunctuationExist) {
                            //System.out.println("puctuation after 1st word exists:  " + punctuationMap.get(countWord) + "\n");
                            words = sb.append(punctuationMap.get(countWord)).toString(); //Insert punctuation after the 1st word.
                        }
                    }

                    if (countWord == indexNum + numberContextWords) { //For last word: should be with punctuation mark before it.
                        boolean ifLastPunctuationExist = IndexMatcher.checkLastPunctuationExist(punctuationMap, countWord);
                        if (ifLastPunctuationExist) {
                            //System.out.println("puctuation before last word exists:  " + punctuationMap.get(countWord-1) + "\n");
                            words = sb.insert(words.lastIndexOf(" "), punctuationMap.get(countWord - 1)).toString();//Insert punctuation before the last one.
                        }
                    }
                }
                System.out.println(words + "\n");
            }

        } else {
            System.out.println("Sorry. No word:'" + wordTarget + "' in this file.");
        }
    }
}
