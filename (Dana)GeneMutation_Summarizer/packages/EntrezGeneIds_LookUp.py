# csvPath = 'gene_results.csv'
import json
import requests
import csv

def convertCsvToDict(csvPath):
    reader = csv.reader(open(csvPath, 'r'))
    dic = {}
    for row in reader:
        geneId, geneName = row
        dic[geneName] = geneId
    return dic

def lookup(csv_to_dic, key):
    if not key:
        print('Please type geneName.')
        return
    if not key in csv_to_dic or not csv_to_dic[key] :
        return
    val = csv_to_dic[key]
    return val

def getGeneId(csv_to_dic, list_geneName):
    dict_geneId_geneName = {}
    for geneName in list_geneName:
        geneId = lookup(csv_to_dic, geneName)
        if geneId:
            dict_geneId_geneName[geneName] = geneId
        else:
            print('No such a gene Id returned by function of getGeneId().')
    print(dict_geneId_geneName)
    return dict_geneId_geneName

    list_geneId = [int(val) for val in dict_geneId_geneName.values()]
