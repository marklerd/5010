# Helper function sheet

# TDM normalising function for test set
TDMSingleWithRef <- function(ref.dt, targ.dt, zero.to.one = TRUE){
  # This function takes array gene expression data.table as a reference and
  # an RNA-seq expression data.table ('target') and returns the TDM
  # normalized, zero to one transformed (if zero.to.one = TRUE) RNA-seq
  # expression data.table
  #
  # Args:
  #   ref.dt: array data.table where the first column contains gene identifiers,
  #           the columns are samples, rows are gene measurements
  #   targ.dt: RNA-seq data.table where the first column contains gene
  #            identifiers, the columns are samples, rows are gene measurements
  #	  zero.to.one: logical - should the data be zero to one transformed?
  #
  # Returns:
  #   tdm.targ: TDM normalized (array data as reference), zero to one
  #                transformed if zero.to.one = TRUE, data.table
  require(data.table)
  # Error-handling
  ref.is.dt <- "data.table" %in% class(ref.dt)
  targ.is.dt <- "data.table" %in% class(targ.dt)
  any.not.dt <- !(any(c(ref.is.dt, targ.is.dt)))
  if (any.not.dt) {
    stop("ref.dt and targ.dt must both be data.tables")
  }
  if (!(all(ref.dt[[1]] %in% targ.dt[[1]]))) {
    stop("Gene identifiers in data.tables must match")
  }
  
  ref.dt <- ensure_numeric_gex(ref.dt)
  targ.dt <- ensure_numeric_gex(targ.dt)
  
  #  message("TDM transformation...\n")
  tdm.targ <- TDM::tdm_transform(target_data = targ.dt,
                                 ref_data = ref.dt,
                                 negative = FALSE,
                                 filter_p = FALSE,
                                 inv_reference = TRUE,
                                 log_target=TRUE)
  
  tdm.targ <- ensure_numeric_gex(tdm.targ)
  
  #  message("\tZero to one transformation...\n")
  if (zero.to.one) {
    tdm.targ <- rescale_datatable(tdm.targ)
  }
  return(data.table(tdm.targ))
}

# TDM normalising function for train set
TDMProcessing <- function(array.dt, seq.dt, zero.to.one = TRUE){
  # This function takes array and RNA-seq data in the form of data.table
  # to be 'mixed' (concatenated) and returns one TDM normalized,
  # zero to one transformed (if zero.to.one = TRUE) data.table. The array data
  # is used as the reference distribution.
  #
  # Args:
  #   array.dt: data.table of array data where the first column contains
  #             gene identifiers, the columns are samples,
  #             rows are gene measurements
  #   seq.dt:   data.table of RNA-seq data where the first column contains
  #             gene identifiers, the columns are samples,
  #             rows are gene measurements
  #   zero.to.one: logical - should data be zero to one transformed?
  #
  # Returns:
  #   tdm.cat: TDM normalized, zero to one transformed if zero.to.one = TRUE,
  #            data.table that contains both array and RNA-seq samples
  #
  require(data.table)
  # Error-handling
  array.is.dt <- "data.table" %in% class(array.dt)
  seq.is.dt <- "data.table" %in% class(seq.dt)
  any.not.dt <- !(any(c(array.is.dt, seq.is.dt)))
  if (any.not.dt) {
    stop("array.dt and seq.dt must both be data.tables")
  }
  if (!(all(array.dt[[1]] %in% seq.dt[[1]]))) {
    stop("Gene identifiers in data.tables must match")
  }
  
  array.dt <- ensure_numeric_gex(array.dt)
  seq.dt <- ensure_numeric_gex(seq.dt)
  print(dim(array.dt))
  
  #  message("TDM transformation...\n")
  tdm.seq <- TDM::tdm_transform(target_data = seq.dt,
                                ref_data = array.dt,
                                negative = FALSE,
                                filter_p = FALSE,
                                inv_reference = TRUE,
                                log_target = TRUE)
  # array.dt <- NAToZero(array.dt)
  # tdm.seq <- NAToZero(tdm.seq)
  #  message("\tZero to one transformation...\n")
  
  tdm.seq <- ensure_numeric_gex(tdm.seq)
  
  if (zero.to.one) {
    array.dt <- rescale_datatable(array.dt)
    tdm.seq <- rescale_datatable(tdm.seq)
  }
  #  message("\tConcatenation...\n")
  tdm.cat <- data.table(cbind(array.dt, tdm.seq[, 2:ncol(tdm.seq), with = F]))
  
  return(data.table(tdm.cat))
}

ensure_numeric_gex <- function(input_data) {
  # Numeric version of data.table
  #
  # Ensure gene expression values are numeric in a given data.table
  #
  # input_data: a data.table with gene in the first column and gene
  # expression in the remaining columns
  #
  # returns a data.table with numeric values for the gene expression columns
  
  if ("data.table" %in% class(input_data)) {
    
    # save gene names as character vector
    gene_names <- as.character(input_data[[1]])
    
    # force gene expression values to be numeric
    gex_values <- t(apply(input_data[,-1], 1, as.numeric))
    
    # create data frame of gene names and gene expression values
    # set column names to be same as input data
    return_df <- data.frame(gene_names, gex_values)
    names(return_df) <- names(input_data)
    
    # return as data.table
    return(data.table(return_df))
    
  } else {
    
    stop("\nInput must be a data.table")
    
  }
}

rescale_01 <- function(data_vector){
  # rescale values in a vector to [0,1]
  # Inputs: vector of numeric values
  # Returns: rescaled vector
  
  # if all the values are the same, return 0 vector
  if (check_all_same(data_vector)) {
    return(rep(0, length(data_vector)))
  } else {
    min_value <- min(data_vector)
    max_value <- max(data_vector)
    rescaled_values <- (data_vector - min_value)/(max_value - min_value)
    return(rescaled_values)
  }
}

rescale_datatable <- function(data_table, by_column = FALSE){
  # rescale each row (or column) of a data table to [0,1]
  # applies rescale_01() to each row (or column)
  # Inputs: gene expression data table
  #   first column of input is genes
  #   remaining columns are expression values
  #   by_column = FALSE rescale each row, if TRUE rescale each column
  # Returns: scaled gene expression data table
  
  data_table <- ensure_numeric_gex(data_table)
  
  data_matrix = data.matrix(data_table[, -1, with = F])
  
  # Rescale each row or column [0,1]
  if (by_column) {
    rescaled_data_matrix = apply(data_matrix, 2, rescale_01)
  } else {
    rescaled_data_matrix = t(apply(data_matrix, 1, rescale_01))
  }
  
  # Include gene symbols in result
  result = data.table(data.frame(data_table[,1], rescaled_data_matrix))
  colnames(result) <- colnames(data_table)
  
  result <- ensure_numeric_gex(result)
  
  return(data.table(result))
  
}

check_all_same <- function(x, my_tolerance = 1e-9){
  # This function returns TRUE if all the elements of the vector are the same
  # within a numerical tolerance levels
  # Thank you: https://stackoverflow.com/a/4752834
  #
  # Args:
  #   x: a numeric vector
  #   my_tolerance: how close must two numbers be for them to be considered equal?
  #
  # Returns:
  #   TRUE or FALSE
  if (is.numeric(x) & is.numeric(my_tolerance)) {
    return(all(abs(max(x) - min(x)) < my_tolerance))  
  } else {
    stop("Vector and tolerance given to check_all_same() must be numeric.")
  }
}


# Function to perform t-test for each gene
perform_t_test <- function(gene_list, df1, df2) {
  # Ensure the 'gene' columns are identical in both dataframes
  if (!all(df1$gene == df2$gene)) {
    stop("Gene names do not match between df1 and df2")
  }
  
  # Initialize a result vector to store p-values
  p_values <- numeric(length(gene_list))
  
  # Loop over each gene in the provided gene list
  for (i in seq_along(gene_list)) {
    gene_name <- gene_list[i]
    
    # Check if the gene exists in both dataframes
    if (!(gene_name %in% df1$gene)) {
      warning(paste("Gene", gene_name, "not found in the dataframes. Skipping."))
      p_values[i] <- NA  # Assign NA for missing genes
      next
    }
    
    # Get the index of the gene
    gene_index <- which(df1$gene == gene_name)
    
    # Extract the numeric values for the current gene from both dataframes
    values_df1 <- as.numeric(df1[gene_index, 2:ncol(df1)])
    values_df2 <- as.numeric(df2[gene_index, 2:ncol(df2)])
    
    # Perform the t-test and store the p-value
    t_test_result <- t.test(values_df1, values_df2)
    p_values[i] <- t_test_result$p.value
  }
  
  # Combine the gene names and p-values into a dataframe
  results_df <- data.frame(
    gene = gene_list,
    p_value = p_values,
    stringsAsFactors = FALSE
  )
  
  # Add significance column
  results_df$significance <- ifelse(results_df$p_value < 0.05, "*", "")
  
  # Ensure the order: gene as the first column and p_value as the second
  results_df <- results_df[, c("gene", "p_value", "significance")]
  
  return(results_df)
}


# Mode function: returns the most frequent value
get_mode <- function(x) {
  uniq_vals <- unique(x)
  uniq_vals[which.max(tabulate(match(x, uniq_vals)))]
}

NPNSingleDT <- function(dt, zero.to.one = FALSE){
  # This function takes gene expression data in the form of a data.table
  # and returns a nonparanormal normalized, zero to one transformed
  # (if zero.to.one = TRUE) data.table
  #
  # Args:
  #   dt: data.table where the first column contains gene identifiers,
  #             the columns are samples, rows are gene measurements
  #	  zero.to.one: logical - should the data be zero to one transformed?
  #
  # Returns:
  #   npn.dt: nonparanormal normalized,
  #           if zero.to.one = TRUE zero to one transformed, data.table
  require(data.table)
  dt.is.dt <- "data.table" %in% class(dt)
  if (!(dt.is.dt)) {
    stop("\nInput must be a data.table")
  }
  
  dt <- ensure_numeric_gex(dt)
  
  val <- data.frame(dt[, 2:ncol(dt), with = F])
  val.mat <- data.matrix(val)
  npn.mat <- huge.npn(t(val.mat), npn.func = "shrinkage",
                      npn.thresh = NULL,
                      verbose = FALSE)
  npn.dt <- data.table(cbind(as.character(dt[[1]]), t(npn.mat)))
  colnames(npn.dt) <- chartr(".", "-", colnames(dt))
  
  npn.dt <- ensure_numeric_gex(npn.dt)
  
  # npn.dt <- NAToZero(npn.dt)
  #  message("\tZero to one transformation...\n")
  if (zero.to.one) {
    npn.dt <- rescale_datatable(npn.dt)
  }
  return(data.table(npn.dt))
}

NPNProcessing <- function(array.dt, seq.dt, zero.to.one = FALSE){
  # This function takes array and RNA-seq data in the form of data.table
  # to be 'mixed' (concatenated) and returns one nonparanormal normalized,
  # zero to one transformed (if zero.to.one = TRUE) data.table
  #
  # Args:
  #   array.dt: data.table of array data where the first column contains
  #             gene identifiers, the columns are samples,
  #             rows are gene measurements
  #   seq.dt:   data.table of RNA-seq data where the first column contains
  #             gene identifiers, the columns are samples,
  #             rows are gene measurements
  #   zero.to.one: logical - should data be zero to one transformed?
  #
  # Returns:
  #   npn.cat: NPN normalized, zero to one transformed if zero.to.one = TRUE,
  #            data.table that contains both array and RNA-seq samples
  #
  require(data.table)
  # Error-handling
  array.is.dt <- "data.table" %in% class(array.dt)
  seq.is.dt <- "data.table" %in% class(seq.dt)
  any.not.dt <- !(any(c(array.is.dt, seq.is.dt)))
  if (any.not.dt) {
    stop("array.dt and seq.dt must both be data.tables")
  }
  if (!(all(array.dt[[1]] %in% seq.dt[[1]]))) {
    stop("Gene identifiers in data.tables must match")
  }
  
  array.dt <- ensure_numeric_gex(array.dt)
  seq.dt <- ensure_numeric_gex(seq.dt)
  
  ref.values <- data.frame(array.dt[, 2:ncol(array.dt), with = F])
  target.values <- data.frame(seq.dt[, 2:ncol(seq.dt), with = F])
  npn.ref <- data.matrix(ref.values)
  npn.array <- huge::huge.npn(t(npn.ref), npn.func = "shrinkage",
                              npn.thresh = NULL, verbose = FALSE)
  npn.targ <- data.matrix(target.values)
  npn.seq <- huge::huge.npn(t(npn.targ), npn.func = "shrinkage",
                            npn.thresh = NULL, verbose = FALSE)
  #  message("\tConcatenation...\n")
  npn.cat <- data.table(cbind(as.character(array.dt[[1]]), t(npn.array), t(npn.seq)))
  colnames(npn.cat) <- c("gene",
                         chartr(".", "-", colnames(ref.values)),
                         chartr(".", "-", colnames(target.values)))
  
  npn.cat <- ensure_numeric_gex(npn.cat)
  
  # npn.cat <- NAToZero(npn.cat)
  #  message("\tZero to one transformation...\n")
  if (zero.to.one) {
    npn.cat <- rescale_datatable(npn.cat)
  }
  return(data.table(npn.cat))
}

# Function to create the boxplot
create_boxplot <- function(array.exp, seq.exp, hk_gene_name) {
  # Melt the dataframes
  df1_melted <- melt(array.exp, id.vars = "gene", variable.name = "sample", value.name = "expression")
  df2_melted <- melt(seq.exp, id.vars = "gene", variable.name = "sample", value.name = "expression")
  print(paste("df1_melted:",dim(df1_melted)))
  print(paste("df2_melted:",dim(df2_melted)))
  
  # Add a source column to distinguish between the two dataframes
  df1_melted$source <- "Microarray"
  df2_melted$source <- "RNA-seq"
  
  # Combine both melted dataframes
  combined_df <- rbind(df1_melted, df2_melted)
  print(paste("combined_df:",dim(combined_df)))
  
  # Apply the t-test for each gene
  genes <- unique(array.exp$gene)
  print(paste("length unique genes:",length(genes)))
  t_test_results <- perform_t_test(genes, array.exp, seq.exp)
  print(paste("t_test_result:",dim(t_test_results)))
  
  # Combine p-values with the combined dataframe for labeling in the plot
  combined_df <- left_join(combined_df, t_test_results, by = "gene")
  
  # Replace Ensembl ID with gene names
  combined_df$gene <- factor(combined_df$gene, levels = names(hk_gene_name))
  combined_df$gene <- as.character(combined_df$gene)
  combined_df$gene <- hk_gene_name[combined_df$gene]
  t_test_results$gene <- hk_gene_name[t_test_results$gene]
  
  print(paste("final combined_df:",dim(combined_df)))
  print(head(combined_df))
  print(table(combined_df$source))
  
  
  # Create boxplots with p-values and significance markers
  box_plot <- ggplot(combined_df, aes(x = source, y = expression, fill = source)) +
    geom_boxplot() +
    facet_wrap(~ gene, scales = "free") +
    labs(title = "Boxplots for each gene with t-test p-values", x = "Source", y = "Expression") +
    theme_minimal() +
    geom_text(data = t_test_results, 
              aes(x = 1.5, y = 0.25, 
                  label = paste("p =", signif(p_value, 4), significance)), 
              inherit.aes = FALSE, size = 3, hjust = 0.5, vjust = -1)
  
  return(box_plot)
}


# data qc and preprocessing function
process_microarray_data <- function(array.data, pheno.data,data_path) {
  # Get sample names
  sns <- sampleNames(array.data)
  
  # Get GEO accessions from sample names
  array_geo <- sub("_.*", "", sns)
  
  # Get GEO accessions from pheno.data
  pheno_geo <- pheno.data$geo_accession
  
  # Ensure GEO accessions match
  if (!identical(array_geo, pheno_geo)) {
    stop("Sample IDs in array.data and pheno.data do not match.")
  }
  
  # Create a new column for diagnostic subtype
  pheno.data <- pheno.data %>%
    mutate(dx_subt = case_when(
      `pregnancy_condition:ch1` == "Normal" ~ NA_real_,  # Return numeric NA
      TRUE ~ as.numeric(`gestation_weeks:ch1`)  # Convert to numeric
    )) %>%
    mutate(dx_subt = case_when(
      dx_subt < 34 ~ "early-onset",
      dx_subt >= 34 ~ "late-onset",
      TRUE ~ "normal"
    ))
  
  # Replace values using case_when
  pheno.data <- pheno.data %>%
    mutate(`pregnancy_condition:ch1` = case_when(
      `pregnancy_condition:ch1` == "Normal" ~ "control",
      `pregnancy_condition:ch1` == "Preeclampsia" ~ "pre-eclampsia",
      TRUE ~ `pregnancy_condition:ch1`  # Keep other values unchanged
    ))
  
  # Extract necessary columns from pheno.data
  hybd_date <- pheno.data$`microarray_hybridization_date:ch1`
  preg_outcome <- pheno.data$`pregnancy_condition:ch1`
  dx_subt <- pheno.data$dx_subt
  
  # Create metadata info dataframe
  info <- data.frame(
    hybd_date = hybd_date,
    disease_status = preg_outcome,
    dx_subt = dx_subt
  )
  rownames(info) <- sns
  
  # Create metadata description
  metadata <- data.frame(
    labelDescription = c("Hybridization Date", "Pregnancy Outcome", "Diagnostic Subtype")
  )
  
  # Convert to AnnotatedDataFrame
  pd <- new("AnnotatedDataFrame", data = info, varMetadata = metadata)
  
  # Assign to the AffyFeatureSet object
  phenoData(array.data) <- pd
  
  # Perform QC of preprocessed data
  # arrayQualityMetrics(array.data,
  #                     outdir = file.path("QC_Report/preprocess",data_path),
  #                     force = TRUE,
  #                     do.logtransform = TRUE,
  #                     intgroup = c("hybd_date"))
  
  # Perform RMA normalization on the raw data
  array.data <- rma(array.data)
  
  # Extract the normalized expression matrix
  array.exp <- exprs(array.data)
  
  # Perform QC of post-processed data
  # arrayQualityMetrics(array.data,
  #                     outdir = file.path("QC_Report/postprocess",data_path),
  #                     force = TRUE,
  #                     do.logtransform = TRUE,
  #                     intgroup = c("hybd_date"))
  
  # Return processed array.data and expression matrix
  return(list(array.data = array.data, array.exp = array.exp))
}



# Function to calculate the CV for each a_value and find the one with the lowest CV for each B
find_lowest_cv <- function(df, numerical_df) {
  
  # Function to calculate coefficient of variation (CV = sd/mean)
  cv <- function(x) {
    return(sd(x) / mean(x))
  }
  
  # Initialize an empty dataframe to store the results
  result_df <- data.frame(ensembl_id = character(),
                          probe_id_with_lowest_CV = numeric(),
                          CV = numeric(),
                          stringsAsFactors = FALSE)
  
  # Iterate over each unique B
  for (b in unique(df$ensembl_id)) {
    a_values <- df$probe_id[df$ensembl_id == b]
    
    # If more than one A corresponds to B, calculate the CV for each A
    if (length(a_values) > 1) {
      cv_list <- sapply(a_values, function(a_value) {
        # Get the row corresponding to the current A from the numerical dataframe
        row_data <- numerical_df[as.character(a_value), ]
        # Calculate the CV for the row (across all columns)
        return(cv(as.numeric(row_data)))
      })
      
      # Find the A with the lowest CV
      min_cv_a_value <- a_values[which.min(cv_list)]
      min_cv_value <- min(cv_list)
      
      # Add the result to the result_df dataframe
      result_df <- rbind(result_df, data.frame(ensembl_id = b, 
                                               probe_id_with_lowest_CV = min_cv_a_value, 
                                               CV = min_cv_value))
    }
  }
  
  # Return the result as a dataframe
  return(result_df)
}

# Function to find the rowname (A) with the highest mean difference between case and control datasets
find_highest_mean_diff <- function(df, case_df, control_df) {
  
  # Initialize an empty dataframe to store the results
  result_df <- data.frame(ensembl_id = character(),
                          probe_id_with_highest_diff = numeric(),
                          Mean_Difference = numeric(),
                          stringsAsFactors = FALSE)
  
  # Iterate over each unique B
  for (b in unique(df$ensembl_id)) {
    a_values <- df$probe_id[df$ensembl_id == b]
    
    # Calculate the mean difference for each A between case and control
    if (length(a_values) > 1) {
      mean_diff_list <- sapply(a_values, function(a_value) {
        # Extract the row for the current A from both case and control datasets
        case_row <- case_df[as.character(a_value), ]
        control_row <- control_df[as.character(a_value), ]
        
        # Calculate the mean difference between case and control for this A
        return(mean(as.numeric(case_row)) - mean(as.numeric(control_row)))
      })
      
      # Find the A with the highest mean difference
      max_diff_a_value <- a_values[which.max(mean_diff_list)]
      max_mean_diff <- max(mean_diff_list)
      
      # Add the result to the result_df dataframe
      result_df <- rbind(result_df, data.frame(ensembl_id = b, 
                                               probe_id_with_highest_diff = max_diff_a_value, 
                                               Mean_Difference = max_mean_diff))
    }
  }
  
  # Return the result as a dataframe
  return(result_df)
}

compress_means_by_ensembl <- function(compressed_gene_mapper, train_df) {
  # Preallocate a list to store results
  result_list <- vector("list", length = nrow(compressed_gene_mapper))
  
  # Iterate through each row of compressed_gene_mapper
  for (i in seq_len(nrow(compressed_gene_mapper))) {
    # Get the current ensembl_id
    ensembl_id <- compressed_gene_mapper$ensembl_id[i]
    
    # Get the list of probe_ids for this ensembl_id, ensure it's not NULL or NA
    probe_ids <- unlist(compressed_gene_mapper$probe_id[i])

    # Use match() to get the row indices that match the probe_ids
    row_indices <- match(probe_ids, rownames(train_df))
    
    # Filter the train_df by valid indices (removing NAs from match)
    valid_indices <- row_indices[!is.na(row_indices)]
    probe_data <- train_df[valid_indices, , drop = FALSE]
    
    # Check if probe_data is valid (non-NULL, and at least one matching row)
    if (nrow(probe_data) > 0) {
      # Calculate the mean for each column (across the probes)
      mean_values <- colMeans(probe_data, na.rm = TRUE)
      
      # Store the result as a list (faster than appending rows)
      result_list[[i]] <- c(ensembl_id, mean_values)
    } else {
      print(paste(ensembl_id, "has no match"))
    }
  }
  
  # Combine the results into a dataframe (filter out NULLs from the list)
  result_df <- do.call(rbind, result_list[!sapply(result_list, is.null)])
  
  # Set column names for the result dataframe (first column is ensembl_id, rest are from train_df)
  colnames(result_df) <- c("ensembl_id", colnames(train_df))
  
  # Convert to a dataframe
  result_df <- as.data.frame(result_df, stringsAsFactors = FALSE)
  
  return(result_df)
}

