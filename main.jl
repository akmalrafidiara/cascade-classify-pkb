using Statistics, DataFrames, Serialization, Distances

include("function.jl")

df = deserialize("data_9m.mat")

function capsul(df)

  # display data
  println("=== Data: ===")
  display(df)

  # compute mean vector
  class = unique(df[:, end])
  class_index = size(df, 2)
  mu_matrix = compute_mean(df, class, class_index)
  println("\nMu per feature per class:")
  display(mu_matrix)
  println()

  # compare values
  class_to_int = Dict(class[i] => i for i in eachindex(class))
  result_df = compare_values(df, mu_matrix, class_index, class_to_int)
  println("\nCalculate the prediction class per feature:")
  display(result_df)
  println()

  # compare labels
  comparison = compare_labels(result_df, df, class_index)
  println("\nComparison of data with label (true if same as class):")
  display(comparison)
  println()

  # calculate accuracy
  println("\n=== Accuracy per feature: ===")
  accuracies = calculate_accuracy(comparison)
  for (i, accuracy) in enumerate(accuracies)
    println("Feature $i Accuracy: $(accuracy * 100)%")
  end

  return accuracies, comparison
end

accuracy, compare = capsul(df)

# sort accuracy
indices = sortperm(accuracy, rev=true)

println("\nSorted Accuracy feature:")
display(indices)

function process_df(df, compare, indices)
  df_current = df
  compare_current = compare

  # 3x4 matrix
  result_matrix = Matrix{Float64}(undef, 3, 0)

  for i in indices
    df_true = df_current[compare_current[:, i].==true, :]
    df_current = df_current[compare_current[:, i].==false, :]
    compare_current = compare_current[compare_current[:, i].==false, :]

    dft = calculate_mean_per_class(df_true, i)
    println("\nSize True Data on Feature $(i) is $(size(df_true, 1))")
    println("Remainder of data: $(size(df_current, 1))")
    display(dft)

    # Append dft
    result_matrix = hcat(result_matrix, dft[:, 2])
  end

  return result_matrix
end

result_matrix = process_df(df, compare, indices)
sorted_indices = sortperm(indices)
result_matrix = result_matrix[:, sorted_indices]
result_matrix = convert(Matrix{Float16}, result_matrix)

println("\nResult of Mu:")
display(result_matrix)