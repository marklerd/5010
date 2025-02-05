library(edgeR)
library(limma)
library(ggplot2)
library(sva)  # Optional, if you want to try ComBat
library(stats)  # For PCA
library(GEOquery)
library(tidyverse)
library(caret)
library(dplyr)
library(genefilter)

# clear environment
rm (list = ls())

# setwd
setwd("~/Documents/PHM5010")

# load helper function file
source("helper_function.R")

# Process rna-seq files
# Get series matrix file and extract metadata
pheno.data <- as.data.frame(pData(getGEO("GSE192902", GSEMatrix = TRUE)[[1]]))

# Load SRA metadata
sra.data <- read.csv("~/Downloads/SraRunTable.txt", sep = ",", header = TRUE, stringsAsFactors = FALSE)

# Proceed with selection and renaming if the column names are correct
sra.data <- sra.data %>%
  as_tibble() %>%  # Ensure compatibility with dplyr functions
  dplyr::select(Sample.Name, gestational_age_in_weeks_at_delivery, gestational_age_in_weeks_at_preeclampsia_onset, Sequencing_batch) %>%
  dplyr::rename(
    sample_id = Sample.Name,
    ga_deliv = gestational_age_in_weeks_at_delivery,
    ga_pre_onset = gestational_age_in_weeks_at_preeclampsia_onset,
    batch = Sequencing_batch
  )

# Merge pheno.data and sra.data by sample_id
pheno.data <- merge(pheno.data, sra.data, by.x = "title", by.y = "sample_id", all = FALSE)

# Prefilter missing GA cases, post partum and ga >= 23 wk cases = 21 early onset, 87 ;ate onset and 296 normal
#control = 238, case = 95
# there are 9 case with unknown ga --> out of 95 cases

#  121
pheno.data <- pheno.data %>%
  filter(`sampling time group:ch1` != "Post-partum"  ) %>%
  filter(`disease:ch1` == "control" | complete.cases(ga_pre_onset))

pheno.data <- pheno.data %>%
  filter(`sampling time group:ch1` != "Post-partum" & `sampling time group:ch1` != "≥23 weeks gestation") %>%
  filter(`disease:ch1` == "control" | complete.cases(ga_pre_onset))

# add disease subtype
# New column for disease subtype: normal == 0, early-onset == 1, late-onset ==2
pheno.data <- pheno.data %>%
  mutate(dx_subt = case_when(
    ga_pre_onset < 34 ~ "early-onset",
    ga_pre_onset >= 34 ~ "late-onset",
    TRUE ~ "normal"
  ))

# pheno metadata column filter and rename
pheno_column <- c("title","geo_accession","cohort:ch1","disease:ch1","sampling time group:ch1","ga_deliv","ga_pre_onset","batch","dx_subt")
pheno.data <- pheno.data[,pheno_column]
colnames(pheno.data) <- c("sample_id","geo_accession","cohort","disease_status","ga_blood_draw","ga_deliv","ga_pre_onset","batch","dx_subt")
# Map "disease_status" into preeclampsia == 1, normal == 0
pheno.data$disease_status <- ifelse(pheno.data$disease_status == "control","control","pre-eclampsia")

# splitting data
# test pre split proportion
pre_split_plot <- ggplot(pheno.data, aes(x = disease_status)) +
  geom_bar(fill="skyblue", color="black") +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  labs(title = "Class Distribution in Train and Test Sets (RNA_seq)",
       x = "Pregnancy outcome",
       y = "Number of cases") +
  theme_minimal()
print(pre_split_plot)
ggsave(filename = "rna_seq_pre-split_proportion.png", plot = pre_split_plot, 
       path = "~/Documents/PHM5010",
       width = 6, height = 4, dpi = 300)
# 180 control 60 cases
rm(pre_split_plot)

# set seed
set.seed(0)
y <- pheno.data$disease_status
trainIndex <- createDataPartition(y,
                                   p=0.7,
                                   list=FALSE)

# Train and test sets
pheno.data_train <- pheno.data[trainIndex, ]
pheno.data_test <- pheno.data[-trainIndex, ]
rm(y,trainIndex)

# Combine the datasets and add a 'set' column
pheno.data_train$set <- "Train"
pheno.data_test$set <- "Test"
combinedData <- rbind(pheno.data_train, pheno.data_test)

# test post split proportion
post_split_plot <- ggplot(combinedData, aes(x = disease_status, fill = set)) + 
  geom_bar(position = "dodge", color = "black") + 
  geom_text(stat='count', aes(label=..count..), position=position_dodge(width=0.9), vjust=-0.5) + 
  labs(title = "Class Distribution in Train and Test Sets", 
       x = "Pregnancy outcome", 
       y = "Number of cases") + 
  theme_minimal()
print(post_split_plot)
ggsave(filename = "rna_seq_post_split_proportion.png", plot = post_split_plot, 
       path = "~/Documents/PHM5010", width = 6, height = 4, dpi = 300)
# train = 126,42
# test = 54,18
rm(combinedData,post_split_plot)

# load expression data
# read files
train_df <- read_csv("~/Documents/PHM5010/GSE192902/GSE192902_counts_Discovery_postQC (3).csv")
head(train_df)

test_1_df <- read_csv("~/Documents/PHM5010/GSE192902/GSE192902_counts_Validation1_postQC.csv")
head(test_1_df)

test_2_df <- read_csv("~/Documents/PHM5010/GSE192902/GSE192902_counts_Validation2_postQC.csv")
head(test_2_df)

# merge post qc dataset
test_1_df <- test_1_df[,c(-1)]
test_2_df <- test_2_df[,c(-1)]
seq.exp <- merge(train_df,test_1_df,by="gene_num",all=TRUE)
seq.exp <- merge(seq.exp,test_2_df,by="gene_num",all=TRUE)
rm(train_df,test_1_df,test_2_df)
write_csv(seq.exp[,c("gene_name","gene_num")],file = "rna_seq_genes.csv")

# split exp data according to train test allocation
train_sample <- pheno.data_train$sample_id
train_sample <- append(c("gene_num","gene_name"),train_sample)
test_sample <- pheno.data_test$sample_id
test_sample <- append(c("gene_num","gene_name"),test_sample)
# seq_sample <- pheno.data$sample_id
# seq_sample <- append(c("gene_num","gene_name"),seq_sample)

# # Train and test sets
train_seq.exp <- seq.exp[,train_sample]
test_seq.exp <- seq.exp[,test_sample]
#seq.exp <- seq.exp[,seq_sample]


# Normalisation and batch removal
process_and_visualize_data <- function(counts, meta, set) {
  
  # Input expression data, metadata, set label
  # Output TMM-normalized post QC expression, log_counts_no_batch
  
  # Ensure the 'batch' column exists in the metadata and is a factor
  if (!"batch" %in% colnames(meta)) {
    stop("Metadata must contain a 'batch' column.")
  }
  
  # Prepare count matrix (assumes first two columns in counts are gene identifiers)
  counts_matrix <- as.matrix(counts[, -c(1, 2)])  # Adjust column indices if needed
  rownames(counts_matrix) <- counts[, 1] # Set rownames to gene IDs
  
  # Create DGEList object
  dge_list <- DGEList(counts = counts_matrix)
  
  # TMM normalization
  dge_list <- calcNormFactors(dge_list, method = "TMM")
  
  # Extract batch information from metadata
  batch <- as.factor(meta$batch)
  
  # Normalize counts to log-CPM (log Counts Per Million)
  log_counts <- cpm(dge_list, log = TRUE)
  
  # ---------------------------- Batch Effect Visualization (Pre-Removal) ----------------------------
  
  # PCA on log-CPM counts before batch effect removal
  pca_pre <- prcomp(t(log_counts), scale. = TRUE)
  
  # PCA plot before batch effect removal
  pca_pre_plot <- ggplot(as.data.frame(pca_pre$x), aes(x = PC1, y = PC2, color = as.factor(batch))) +
    geom_point(size = 3) +
    labs(title = "PCA - Pre Batch Effect Removal", x = "PC1", y = "PC2") +
    theme_minimal() +
    scale_color_discrete(name = "Batch")
  
  # Save the PCA plot (pre-batch effect removal)
  ggsave(filename = paste0(set, "_seq_PCA_pre_batch_removal.png"), plot = pca_pre_plot, width = 7, height = 5)
  
  # ----------------------------- Batch Effect Removal using limma -----------------------------
  
  # Remove batch effect using limma
  log_counts_no_batch <- removeBatchEffect(log_counts, batch = batch)
  
  # ------------------------ PCA Visualization (Post-Batch Effect Removal) ---------------------
  
  # PCA on log-CPM counts after batch effect removal
  pca_post <- prcomp(t(log_counts_no_batch), scale. = TRUE)
  
  # PCA plot after batch effect removal
  pca_post_plot <- ggplot(as.data.frame(pca_post$x), aes(x = PC1, y = PC2, color = as.factor(batch))) +
    geom_point(size = 3) +
    labs(title = "PCA - Post Batch Effect Removal", x = "PC1", y = "PC2") +
    theme_minimal() +
    scale_color_discrete(name = "Batch")
  
  # Save the PCA plot (post-batch effect removal)
  ggsave(filename = paste0(set, "_seq_PCA_post_batch_removal.png"), plot = pca_post_plot, width = 7, height = 5)
  
  # Return the DGEList object and the logCPM matrix with batch effect removed
  return(list(dge_list$samples,log_counts_no_batch))
}

# visualise and save normalised data
train_results <- process_and_visualize_data(train_seq.exp, pheno.data_train, "train")
test_results <- process_and_visualize_data(test_seq.exp, pheno.data_test, "test")
#seq.data <- process_and_visualize_data(seq.exp,pheno.data, "Seq")

seq_ensembl_ids <- rownames(train_results[[2]])

#seq_ensembl_ids <- rownames(seq.data[[2]])
# length(seq_ensembl_ids) = 7160

ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
gene_biot = getBM(attributes = c('ensembl_gene_id', 'gene_biotype','chromosome_name'), 
                  filters = 'ensembl_gene_id', 
                  values = seq_ensembl_ids, 
                  mart=ensembl)
pc_seq_ensembl_ids <- gene_biot %>%
  filter(gene_biotype == "protein_coding") %>%
  #filter(chromosome_name %in% as.character(1:22)) %>%
  pull(ensembl_gene_id)
# length(pc_seq_ensembl_ids) = 6716


train_seq.exp <- as.data.frame(train_results[[2]]) # dim(train_seq.exp) = 7160  168
pc_train_seq.exp <- as.data.frame(train_seq.exp[rownames(train_seq.exp) %in% pc_seq_ensembl_ids,]) # dim(pc_train_seq.exp) = 6716  168
test_seq.exp <- as.data.frame(test_results[[2]]) # dim(test_seq.exp) = 7160   72
pc_test_seq.exp <- as.data.frame(test_seq.exp[rownames(test_seq.exp) %in% pc_seq_ensembl_ids,]) # dim(pc_test_seq.exp) = 6716   72

#
#processed_seq.exp <- as.data.frame(seq.data[[2]])
#pc_processed_seq.exp <- as.data.frame(processed_seq.exp[rownames(processed_seq.exp) %in% pc_seq_ensembl_ids,])
#dim(pc_processed_seq.exp) = 6716  240

#processed_seq.exp <- rownames_to_column(processed_seq.exp, var = "gene")
#pc_processed_seq.exp <- rownames_to_column(pc_processed_seq.exp, var = "gene")
train_seq.exp <- rownames_to_column(train_seq.exp, var = "gene")
pc_train_seq.exp <- rownames_to_column(pc_train_seq.exp, var = "gene")
test_seq.exp <- rownames_to_column(test_seq.exp, var = "gene")
pc_test_seq.exp <- rownames_to_column(pc_test_seq.exp, var = "gene")


pheno.data_train <- pheno.data_train[,c("sample_id","disease_status","dx_subt")]
pheno.data_test <- pheno.data_test[,c("sample_id","disease_status","dx_subt")]
#pheno.data <- pheno.data[,c("sample_id","disease_status","dx_subt")]


# Transpose the dataframe and convert it to a dataframe
t_train_seq.exp <- as.data.frame(t(train_seq.exp[, -1]))
t_pc_train_seq.exp <- as.data.frame(t(pc_train_seq.exp[, -1]))
t_test_seq.exp <- as.data.frame(t(test_seq.exp[, -1]))
t_pc_test_seq.exp <- as.data.frame(t(pc_test_seq.exp[, -1]))
#t_processed_seq.exp <- as.data.frame(t(processed_seq.exp[, -1]))
#t_pc_processed_seq.exp <- as.data.frame(t(pc_processed_seq.exp[, -1]))


# Assign new column names
colnames(t_train_seq.exp) <- train_seq.exp$gene
colnames(t_pc_train_seq.exp) <- pc_train_seq.exp$gene
colnames(t_test_seq.exp) <- test_seq.exp$gene
colnames(t_pc_test_seq.exp) <- pc_test_seq.exp$gene

#colnames(t_processed_seq.exp) <- processed_seq.exp$gene
#colnames(t_pc_processed_seq.exp) <- pc_processed_seq.exp$gene

write_csv(t_train_seq.exp,file = "seq_exp_train_all.csv")
write_csv(t_pc_train_seq.exp,file = "seq_exp_train_pc.csv")
write_csv(t_test_seq.exp,file = "seq_exp_test_all.csv")
write_csv(t_pc_test_seq.exp,file = "seq_exp_test_pc.csv")
write_csv(pheno.data_train, file = "seq_pheno_train.csv")
write_csv(pheno.data_test, file = "seq_pheno_test.csv")




write_csv(t_processed_seq.exp,file = "seq_exp_all_genes.csv")
write_csv(t_pc_processed_seq.exp, file = "seq_exp_pc_genes.csv")
write_csv(pheno.data, file = "seq_pheno.csv")

# Save TMM-normalized samples
#write.csv(train_results[[1]], file = "train_TMM_postQC.csv")
#write.csv(test_results[[1]], file = "test_TMM_postQC.csv")
write.csv(seq.data[[1]], file = "seq_TMM_postQC.csv")

# Save batch-effect removed log-CPM counts (optional)
#write.csv(train_results[[2]], file = "train_logCPM_postQC_RemovedBatch.csv")
#write.csv(test_results[[2]], file = "test_logCPM_postQC_RemovedBatch.csv")
write.csv(seq.data[[2]], file = "seq_logCPM_postQC_RemovedBatch.csv")

# save pheno.data
saveRDS(pheno.data_train, file = "rna_seq_train_pheno.rds")
saveRDS(pheno.data_test, file = "rna_seq_test_pheno.rds")
