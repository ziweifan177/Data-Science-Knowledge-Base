/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package LogicPackage;

import OperationPackage.IndexMatcher;
import OperationPackage.TreemapIterator;
import OperationPackage.WordLocator;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.NavigableSet;
import java.util.TreeMap;

public class SearchEngine {

    /** Return the index/key list of target word: */
    public static ArrayList search(TreeMap map, String word) {
        ArrayList indexList = WordLocator.wordLocate(map, word);
        ArrayList<Integer> wordIndexList = new ArrayList<Integer>(indexList.size());

        for (Object index : indexList) {
            wordIndexList.add(Integer.valueOf(index.toString()));
        }
        return wordIndexList;
    }
}
