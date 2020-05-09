csvPath = 'gene_results.csv'
#--------------------------
#Mutation URL:
URL_base = "https://www.cbioportal.org/api/molecular-profiles/"
branch_mutations = "gbm_tcga_mutations/mutations/"
para_mutations = "fetch?projection=SUMMARY&pageSize=10000000&pageNumber=0&direction=ASC"
#------------------------
#Copy Num URL:
branch_copynum = "gbm_tcga_gistic/discrete-copy-number/"
para_copynum = "fetch?discreteCopyNumberEventType=ALL&projection=SUMMARY"
#------------------------
#Patiens URL:
URL_patients = "https://www.cbioportal.org/api/studies/gbm_tcga/patients/"
#-------------------------
global sampleListId
sampleListId = 'gbm_tcga_cnaseq'
#---------------------
#---------------------
URL_mutation = URL_base + branch_mutations + para_mutations
URL_copynum = URL_base + branch_copynum + para_copynum
size_patients = 0
