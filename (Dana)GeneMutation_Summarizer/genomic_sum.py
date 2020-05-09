import json
import requests
import csv
from packages import EntrezGeneIds_LookUp
from packages import RESTprocessor
from packages import calculator
import parameters
import sys

# Initialize variables:
list_geneId = []
argv = sys.argv #list_geneName = ['TP53','MDM2','MDM4']
list_geneName = argv[1:]

if (not list_geneName or len(list_geneName)>3):
    print("Please input geneName. Up to 3.")

else:
# Convert .csv to hash for quick lookup.
    csv_to_dic = EntrezGeneIds_LookUp.convertCsvToDict(parameters.csvPath)

# Lookup geneId according to geneNames.
    dict_geneId_geneName = EntrezGeneIds_LookUp.getGeneId(csv_to_dic, list_geneName)
    list_geneId = [int(val) for val in dict_geneId_geneName.values()]

# Get patients numbers(cases)
    size_patients = len(RESTprocessor.GETRequester(parameters.URL_patients))
    parameters.size_patients = size_patients

# Calculate: mutation rate, copy number altered rate.
    dict_mutations = calculator.getGroupMutationCount(list_geneId)
    group_mutation_rates = calculator.getGroupMutationRate(dict_mutations)

    dict_countCopyNum = calculator.getCopyNum(parameters.URL_copynum, list_geneId)
    group_CopyNumAlteredRates = calculator.getGroupCopyNumAlteredRate(dict_countCopyNum, parameters.size_patients)
#---------------------------------------------
# OUTPUT:
# Check the GeneName according to GeneId
    def checkGeneNames(dict_geneId_geneName, geneId):
        for name, gid in dict_geneId_geneName.items():
            if int(gid)== int(geneId):
                return name

    if len(group_mutation_rates)!= len(group_CopyNumAlteredRates):
        print("Error in length.")

    length = len(group_CopyNumAlteredRates)
    if length==0:
        print ("No Gene.")

    elif length == 1:
        geneName = checkGeneNames(dict_geneId_geneName, list(group_mutation_rates.keys())[0])
        print(geneName, 'is mutated in ', list(group_mutation_rates.values())[0], '% of all cases.')

        print(geneName, ' is copy number altered in ', list(group_CopyNumAlteredRates.values())[0], '% of all cases.')

        comb_rate = calculator.combMutationAndCopyNum(group_mutation_rates, group_CopyNumAlteredRates)
        print('Cases with at least one mutation or copy number alteration in ', geneName, ': ', list(comb_rate.values())[0], '% of all cases.')

    elif length >= 1:
        comb_rate = calculator.combMutationAndCopyNum(group_mutation_rates, group_CopyNumAlteredRates)
        total_rate = 0
        for geneId, rate in comb_rate.items():
            geneName = checkGeneNames(dict_geneId_geneName, geneId)
            print(geneName, 'is mutated or copy number altered in ', rate ,'% of cases.')
            total_rate += rate
        print('Cases with at least one mutation or copy number alteration in one of the genes:', round(total_rate), '%')
