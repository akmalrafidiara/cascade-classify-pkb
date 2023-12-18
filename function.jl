function compute_mean(df, class, class_index)
  mu_matrix = zeros(Float16, length(class), class_index - 1)

  for (i, c) in enumerate(class)
    current_class_pos = (df[:, class_index] .== c)
    current_df = Float32.(df[current_class_pos, 1:class_index-1])
    mu_matrix[i, :] = mean(current_df, dims=1)
  end

  return mu_matrix
end

function compare_values(df, mu_matrix, class_index, class_to_int)
  result_df = Matrix{Float16}(undef, size(df, 1), class_index - 1)

  for i in eachindex(df[:, 1])
    for j in eachindex(1:class_index-1)
      min_distance = Inf
      min_class = nothing

      for (class, class_int) in class_to_int
        distance = abs(mu_matrix[class_int, j] - df[i, j])

        if distance < min_distance
          min_distance = distance
          min_class = class
        end
      end

      result_df[i, j] = class_to_int[min_class]
    end
  end

  return result_df
end

function compare_labels(result_df, df, class_index)
  comparison = Matrix{Bool}(undef, size(result_df, 1), size(result_df, 2))

  for i in eachindex(result_df[:, 1])
    for j in eachindex(result_df[1, :])
      comparison[i, j] = result_df[i, j] == df[i, class_index]
    end
  end

  return comparison
end

function calculate_accuracy(comparison)
  accuracy = zeros(Float32, size(comparison, 2))

  for i in eachindex(comparison[1, :])
    accuracy[i] = mean(comparison[:, i])
  end

  return accuracy
end

function calculate_mean_per_class(data, col)
  classes = unique(data[:, end])
  means = Matrix{Float64}(undef, length(classes), 2)

  for (i, class) in enumerate(classes)
    class_data = data[data[:, end].==class, :]
    class_mean = mean(Float64.(class_data[:, col]))
    means[i, 1] = class
    means[i, 2] = class_mean
  end

  return means
end