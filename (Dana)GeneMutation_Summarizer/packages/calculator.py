import json
import requests
import csv
import parameters
from packages import RESTprocessor

def getGroupMutationCount(list_geneId):
    dict_mutations = {}
    json_={"entrezGeneIds":list_geneId,
            "sampleListId": parameters.sampleListId
    }
    size_mutations = RESTprocessor.POSTRequester(parameters.URL_mutation, json_)

    for geneId in list_geneId:
        count = 0
        #print(geneId)

        for item in size_mutations:
            if item['entrezGeneId']== geneId:
                count += 1
                dict_mutations[geneId] = count
    size_mutations = None # Clear cache
    return dict_mutations

def getGroupMutationRate(dict_mutations):
    group_mutation_rates = {}
    for geneid, mutation in dict_mutations.items():
        group_mutation_rates[geneid] = round(mutation/parameters.size_patients,3) * 100
    return group_mutation_rates

def getCopyNum(URL_copynum, list_geneId):
    dict_countCopyNum = {}
    json_={"entrezGeneIds":
        list_geneId,  #geneId=7157,
      "sampleListId": parameters.sampleListId
    }

    json_res = RESTprocessor.POSTRequester(URL_copynum, json_)

    for geneId in list_geneId:
        count = 0
        for item in json_res:
            if item['entrezGeneId']== geneId and item['alteration'] != 0 and item['alteration'] != 'NA':
                count += 1
        dict_countCopyNum[geneId] = count
    json_res = None # Clear cache
    return dict_countCopyNum

def getGroupCopyNumAlteredRate(dict_countCopyNum, size_patients):
    group_CopyNumAlteredRates = {}
    for geneId, copyNum in dict_countCopyNum.items():
        group_CopyNumAlteredRates[geneId] = round(copyNum/size_patients, 3) * 100
    return group_CopyNumAlteredRates

def combMutationAndCopyNum(group_mutation_rates, group_CopyNumAlteredRates):
    new_dict = {}
    for id_mut, rate_mut in group_mutation_rates.items():
        for id_co, rate_co in group_CopyNumAlteredRates.items():
            if id_mut == id_co:
                id_ = id_mut
                new_dict[id_] = round(rate_mut + rate_co, 3)
    return new_dict
