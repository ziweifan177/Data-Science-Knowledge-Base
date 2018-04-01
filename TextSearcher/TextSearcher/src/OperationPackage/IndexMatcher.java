/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package OperationPackage;

import java.util.ArrayList;
import java.util.TreeMap;
import java.util.regex.Pattern;


public class IndexMatcher {

    /**Check if the punctuation exists after the word.  */
    public static boolean checkFirstPunctuationExist(TreeMap mapPunctuations, int indexWord) {
        boolean ifPunctuationExist = false;
        String punc = mapPunctuations.get(indexWord).toString();
        if (Pattern.matches("\\p{Punct}", (CharSequence) mapPunctuations.get(indexWord).toString())) {
            ifPunctuationExist = true;
        }
        return ifPunctuationExist;
    }

    /**Check if the punctuation exists in front of the word.  */
    public static boolean checkLastPunctuationExist(TreeMap mapPunctuations, int indexWord) {
        boolean ifPunctuationExist = false;
        String punc = mapPunctuations.get(indexWord-1).toString();
        if (Pattern.matches("\\p{Punct}", (CharSequence) mapPunctuations.get(indexWord - 1).toString())) {
            ifPunctuationExist = true;
        }
        return ifPunctuationExist;
    }
    
}
