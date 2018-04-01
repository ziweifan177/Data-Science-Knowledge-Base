/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package OperationPackage;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;

public class WordLocator {

    /** Locate one word in one treemap, and return the index in the treemap. */
    public static ArrayList wordLocate(TreeMap map, String word) {
        Set ref = map.keySet();
        Iterator it = ref.iterator();
        ArrayList list = new ArrayList();

        while (it.hasNext()) {
            Object o = it.next();
            if (map.get(o).equals(word)) {
                list.add(o);
            }
        }
        return list;
    }

}
