# Load packages
library(tidyverse)
library(dplyr)
library(affy)            # For handling Affymetrix .CEL files and RMA normalization
library(pd.hugene.1.0.st.v1)  # For annotation
library(annotate)        # For annotation
library(oligoClasses)    # For working with the ExpressionSet
library(affyPLM)         # For quality control of arrays
library(ggplot2)         # For PCA plotting
library(Biobase)         # For merging phenotypes with the ExpressionSet
library(oligo)
library(GEOquery)
library(arrayQualityMetrics)
library(pd.hugene.1.0.st.v1)
library(hugene10sttranscriptcluster.db)
library(AnnotationDbi)  # For annotation functions
library(caret)
library(ggplot2)
library(genefilter)
library(biomaRt)
library(limma)
library(RankProd)

# clear environment
rm (list = ls())

# setwd
setwd("~/Documents/PHM5010")


# load helper function file
source("helper_function.R")

# Process microarray files
# get series matrix file
gse <- getGEO("GSE85307", GSEMatrix = TRUE)

# fetch feature data to get ID - gene symbol mapping
feature.data <- as.data.frame(fData(gse[[2]]))
pheno.data <- as.data.frame(pData(gse[[2]]))
rm(gse)
# 33297 features loaded
# 157 enteries loaded

# splitting data
# test pre split proportion
pre_split_plot <- ggplot(pheno.data, aes(x = `pregnancy_condition:ch1`)) +
  geom_bar(fill="skyblue", color="black") +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  labs(title = "Class Distribution in Train and Test Sets",
       x = "Pregnancy outcome",
       y = "Number of cases") +
  theme_minimal()
 print(pre_split_plot)
ggsave(filename = "pre-split_proportion.png", plot = pre_split_plot, 
       path = "~/Documents/PHM5010",
       width = 6, height = 4, dpi = 300)
rm(pre_split_plot)

# set seed
set.seed(0)
y <- pheno.data$`pregnancy_condition:ch1`
#trainIndex <- createDataPartition(y,
#                                  p=0.7,
#                                  list=FALSE)

# Train and test sets
#pheno.data_train <- pheno.data[trainIndex, ]
#pheno.data_test <- pheno.data[-trainIndex, ]
#rm(y,trainIndex)

# Combine the datasets and add a 'set' column
#pheno.data_train$set <- "Train"
#pheno.data_test$set <- "Test"
#combinedData <- rbind(pheno.data_train, pheno.data_test)

# test post split proportion
post_split_plot <- ggplot(combinedData, aes(x = `pregnancy_condition:ch1`, fill = set)) + 
  geom_bar(position = "dodge", color = "black") + 
  geom_text(stat='count', aes(label=..count..), position=position_dodge(width=0.9), vjust=-0.5) + 
  labs(title = "Class Distribution in Train and Test Sets", 
       x = "Pregnancy outcome", 
       y = "Number of cases") + 
  theme_minimal()
print(post_split_plot)
ggsave(filename = "post_split_proportion.png", plot = post_split_plot, 
       path = "~/Documents/PHM5010", width = 6, height = 4, dpi = 300)
rm(combinedData,post_split_plot)

# Specify the path to your .CEL files
celFiles <- list.celfiles("GSE85307/data", full.names = TRUE, pattern = "\\.CEL$")

#filter train test files
#geo_train <- rownames(pheno.data_train)
#geo_test <- rownames(pheno.data_test)

# regex pattern
#train_pattern <- paste0("(", paste(geo_train, collapse = "|"), ")_")
#test_pattern <- paste0("(", paste(geo_test, collapse = "|"), ")_")

#train_files <- celFiles[grepl(train_pattern, celFiles)]
#test_files <- celFiles[grepl(test_pattern, celFiles)]

#rm(celFiles,geo_test,geo_train,train_pattern,test_pattern)

# Read in the data using oligo
#train.array.data <- read.celfiles(train_files)
#test.array.data <- read.celfiles(test_files)
#rm(train_files,test_files)
array.data <- read.celfiles(celFiles)

# get pheno.data, data qc and normalisation
# process train data
# train.data.p <- process_microarray_data(train.array.data, pheno.data_train, "train")
# train.array.data <- train.data.p$array.data
# train.array.exp <- train.data.p$array.exp
# test.data.p <- process_microarray_data(test.array.data, pheno.data_test, "test")
# test.array.data <- test.data.p$array.data
# test.array.exp <- test.data.p$array.exp
array.data.p <- process_microarray_data(array.data, pheno.data,"microarray")
array.data <- array.data.p$array.data
array.exp <- array.data.p$array.exp

# probe annotation
# get mapper
# gene_mapper <- as.data.frame(hugene10sttranscriptclusterENSEMBL) # 22864 rows
# array_chr <- as.data.frame(hugene10sttranscriptclusterCHR) #19883 rows
# autosomal_probes <- array_chr[array_chr$chromosome %in% as.character(1:22), "probe_id"] #19049
# gene_mapper <- gene_mapper[c(gene_mapper$probe_id %in% autosomal_probes),] #22020 rows -> 3388 probe_id duplicates, 2082 ensembl duplicates

### Get ensembl gene id, ensembl transcript id, chromosome number from feature metadata
extracted_feature.data <- feature.data %>%
  mutate(
    ENST_ids = str_extract_all(mrna_assignment, "ENST\\d{11}"),
    ensembl_id = str_extract_all(mrna_assignment, "ENSG\\d{11}"),
    chr = str_extract(mrna_assignment, "chr(\\d{1,2})") %>% str_remove("chr")  %>% as.numeric()
  ) %>%
  filter(category == "main") %>%
  dplyr::select(gene_assignment, mrna_assignment, ENST_ids, ensembl_id, chr)

# Get unique ensembl gene id and its associated probes = 29932
unique_ensg <- extracted_feature.data %>%
  # Unnest the ENSG_ids column while keeping track of the original row number
  mutate(probe_ids = rownames(extracted_feature.data)) %>%
  # filter autosomal probes
  #filter(chr %in% as.character(1:22)) %>%
  unnest(ensembl_id) %>%
  group_by(ensembl_id) %>%
  reframe(probe_id = list(unique(probe_ids))) %>%
  as.data.frame()


# filter low variable genes
#train.array.exp <- varFilter(train.array.exp,
#                         var.func = IQR,            # Use IQR as the filtering function
#                         var.cutoff = 0.2,          # Set the cutoff low variable genes, keeping the top 80% of most variable genes
#                         filterByQuantile = TRUE)   # Treat cutoff as a quantile
#test.array.exp <- test.array.exp[rownames(test.array.exp) %in% rownames(train.array.exp), ]
# dim(array.exp) pre-filter = 33297
array.exp <- varFilter(array.exp,
                       var.func = IQR,            
                       var.cutoff = 0.2,         
                       filterByQuantile = TRUE)
# dim(array.exp) post-filter = 26637

# filter gene mapper
# Convert to character
# gene_mapper$probe_id <- as.character(gene_mapper$probe_id)
# gene_mapper$ensembl_id <- as.character(gene_mapper$ensembl_id)
# rownames(train.array.exp) <- as.character(rownames(train.array.exp))
# # Trim whitespaces
# gene_mapper$probe_id <- trimws(gene_mapper$probe_id)
# gene_mapper$ensembl_id <- trimws(gene_mapper$ensembl_id)
# rownames(train.array.exp) <- trimws(rownames(train.array.exp))

# Filter rows
# gene_mapper <- gene_mapper[gene_mapper$probe_id %in% rownames(train.array.exp), ]

#######
# There are many duplicates in gene_mapper --> probe_ids and ensembl_id but no duplicated combination
# # Get dataframe of unique ensembl_id in column 1 with duplicated probe_id in a list in column 2
# compressed_gene_mapper <- gene_mapper %>%
#   group_by(ensembl_id) %>%
#   reframe(probe_id = list(probe_id)) %>%
#   as.data.frame()

# Find average of each ensembl id with many probe ids
# compressed_train.array.exp <- compress_means_by_ensembl(unique_ensg,train.array.exp)
# compressed_test.array.exp <- compress_means_by_ensembl(unique_ensg,test.array.exp)
compressed_array.exp <- compress_means_by_ensembl(unique_ensg,array.exp)
# mapped ensembl gene id = 23916

# # get case/ control
# train_case <- rownames(train.array.pheno[train.array.pheno$preg_outcome == "pre-eclampsia",])
# train_cont <- rownames(train.array.pheno[train.array.pheno$preg_outcome == "control",])
# train_case.exp <- compressed_train.array.exp[,train_case]
# train_cont.exp <- compressed_train.array.exp[,train_cont]

# get pheno data for train/ test for saving
#train.array.pheno <- pData(train.array.data)
#test.array.pheno <- pData(test.array.data)
array.pheno <- pData(array.data)

# Example list of ExpressionSet objects
# array_list <- list(
#   train.array.exp = compressed_train.array.exp,
#   array_train_pheno = train.array.pheno,
#   test.array.exp = compressed_test.array.exp,
#   array_test_pheno = test.array.pheno,
#   compressed_gene_mapper = extracted_feature.data
# )
# 
# # Save each ExpressionSet object to a .rds file
# for (name in names(array_list)) {
#   saveRDS(array_list[[name]], file = paste0(name, ".rds"))
# }

ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
array_ensembl_ids <- compressed_array.exp$ensembl_id # length = 23916
gene_biot = getBM(attributes = c('ensembl_gene_id', 'gene_biotype','chromosome_name'), 
                  filters = 'ensembl_gene_id', 
                  values = array_ensembl_ids, 
                  mart=ensembl)
pc_array_ensembl_ids <- gene_biot %>%
  filter(gene_biotype == "protein_coding") %>%
  #filter(chromosome_name %in% as.character(1:22)) %>%
  pull(ensembl_gene_id) # length Protein-coding = 16926

# Rename columns to keep only "GSM" followed by 7 digits, skipping the first column
colnames(compressed_array.exp)[-1] <- gsub("^(GSM[0-9]{7})_.*", "\\1", colnames(compressed_array.exp)[-1])
pc_compressed_array.exp <- as.data.frame(compressed_array.exp[compressed_array.exp$ensembl_id %in% pc_array_ensembl_ids,]) #16926 rows, 157 samples

rownames(array.pheno) <- gsub("^(GSM[0-9]{7})_.*", "\\1", rownames(array.pheno))
array.pheno <- rownames_to_column(array.pheno, var = "sample_id")
array.pheno <- array.pheno[,c("sample_id","disease_status","dx_subt")]

colnames(compressed_array.exp)[1] <- "gene"
colnames(pc_compressed_array.exp)[1] <- "gene"



# Transpose the dataframe and convert it to a dataframe
t_compressed_array.exp <- as.data.frame(t(compressed_array.exp[, -1]))
t_pc_compressed_array.exp <- as.data.frame(t(pc_compressed_array.exp[, -1]))

# Add the new column for IDs
rownames(t_compressed_array.exp) <- colnames(compressed_array.exp)[-1]
rownames(t_pc_compressed_array.exp) <- colnames(pc_compressed_array.exp)[-1]

# Assign new column names
colnames(t_compressed_array.exp) <- compressed_array.exp$gene
colnames(t_pc_compressed_array.exp) <- pc_compressed_array.exp$gene

#dim(t_compressed_array.exp) = 157 23916
# dim(t_pc_compressed_array.exp) = 157 16926

write_csv(t_compressed_array.exp,file = "array_exp_all_genes.csv")
write_csv(t_pc_compressed_array.exp, file = "array_exp_pc_genes.csv")
write_csv(array.pheno, file = "array_pheno.csv")

### Differential expression by limma
#eset <- oligo::rma(array.data)
exprs_data <- compressed_array.exp
rownames(exprs_data) <- exprs_data$ensembl_id
exprs_data$ensembl_id<- NULL
exprs_matrix <- as.matrix(exprs_data)
mode(exprs_matrix) <- "numeric"
eset_gene <- ExpressionSet(assayData = exprs_matrix)
group <- factor(pheno.data$`pregnancy_condition:ch1`, levels = c("Normal", "Preeclampsia"))
design <- model.matrix(~0 + group)
colnames(design) <- levels(group)
#design <- model.matrix(~factor(Groups))
#colnames(design) <- c("Preeclampsia", "PreeclampsiavsNormal")

# Fit the linear model
fit <- lmFit(eset_gene, design)
#fit <- eBayes(fit)
contrast.matrix <- makeContrasts(Preeclampsia - Normal, levels = design)
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

# Extract Differential Expression Results with FDR Correction
results <- topTable(fit2, adjust.method = "fdr", number = Inf)
filtered_results <- subset(results, adj.P.Val < 0.05 & abs(logFC) > log2(1.2))

print(results)
print(filtered_results)

results <- rownames_to_column(results, var = "gene")
  
write_csv(results, file = "array_dge.csv")


# Adjust p-values using FDR correction (set p-value adjustment to FDR)
#options(digits=2)

# Get the results table
#res <- topTable(fit, number=Inf, adjust.method="fdr", coef=1)

# Check the summary of fold change and adjusted p-value before filtering
#summary(res$logFC)  # Check fold change values
#summary(res$adj.P.Val)  # Check adjusted p-values

### RankProd
exprs_data <- pc_compressed_array.exp
rownames(exprs_data) <- exprs_data$ensembl_id
exprs_data$ensembl_id <- NULL
exprs_matrix <- as.matrix(exprs_data)
mode(exprs_matrix) <- "numeric"
group <- factor(pheno.data$`pregnancy_condition:ch1`, levels = c("Normal", "Preeclampsia"))
group_vector <- as.numeric(group) - 1 
# Define group labels (make sure it corresponds to your samples)

# Perform differential expression analysis using RankProd
RP.out <- RankProducts(exprs_matrix, group_vector, logged = TRUE, na.rm = FALSE, plot = FALSE, rand = 123)  # Set num.perm to a suitable value (e.g., 1000)

