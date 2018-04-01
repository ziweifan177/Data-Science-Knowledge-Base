/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package OperationPackage;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeMap;

/**
 *
 * @author Wing Fan
 */
public class TreemapIterator {

    public static boolean containsWord(TreeMap map, String word) {

        boolean ifContains = map.containsValue(word);
        return ifContains;
    }
}
