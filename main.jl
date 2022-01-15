using Random
using CSV, DataFrames, Statistics
using JuMP, CPLEX
using PyCall, PyPlot
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split
@sk_import linear_model: LinearRegression
@sk_import metrics: r2_score

py"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def printCorrelationMatrix(corr):

    sns.set_theme(style="white")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(14, 12))
    ax.set_title('Pearson Correlation')

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, 
                mask=mask, 
                cmap="coolwarm", 
                vmax=1, 
                vmin = -1, 
                center=0.5,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .4},
                annot = True,
                annot_kws = {"size": 13})

def printParetoFront(XX,XY):
    sns.set_theme(style="white",font_scale=3)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(14, 12))
    ax.set_title('Pareto Front')

    d = {'Total X-X correlations': XX, 'Total X-Y correlations': XY}
    df = pd.DataFrame(data=d)
    
    sns.lineplot(data = df, 
                 x='Total X-X correlations', 
                 y='Total X-Y correlations',
                 markers=True,
                 marker='o',
                 markersize = 12)

def printR2(X,Y):
    sns.set_theme(style="whitegrid",font_scale=3)
    d = {'Points': X, 'r2': Y}
    df = pd.DataFrame(data=d)
    f, ax = plt.subplots(figsize=(14, 12))
    sns.barplot(x="Points", y="r2", data=df)
"""

function loadData(dataset)
    df = CSV.read(dataset, DataFrame)
    ρ = cor(Matrix(df))                # Pearson correlation   
    py"printCorrelationMatrix"(ρ)
    display(gcf())
    savefig("results/correlationMatrix.pdf")
    ρ = abs.(cor(Matrix(df)))
    return Matrix(df[:,1:end-1]), Vector(df[:,end]), ρ[1:end-1,1:end-1], ρ[1:end-1,end]
end

mutable struct Solution
    id
    obj
    variables
    θ
    φ
    r2
end

function executeVariableSelectModel(ρij, ρ0i, α)
    
    N = size(ρij,1)

    m = Model(CPLEX.Optimizer)

    @variable(m, u[i in 1:N], Bin)
    @variable(m, v[i in 1:N, j in 1:N; i<j])

    @objective(m, Max, α * sum( abs(ρ0i[i]) * u[i] for i in 1:N ) - (1-α) * sum( abs(ρij[i,j]) * v[i,j] for i in 1:N-1, j in i+1:N))

    for i in 1:N-1
        for j in i+1:N
            @constraint(m, v[i,j] >= 0)
            @constraint(m, v[i,j] >= u[i] + u[j] - 1)
        end
    end

    set_silent(m)

    optimize!(m)
    u_opt = JuMP.value.(u)
    v_opt = JuMP.value.(v)
    obj = JuMP.objective_value(m)

    φ = 0.0
    M = zeros(Bool,N)
    for i in 1:N-1
        for j in i+1:N
            if v_opt[i,j] > 0.5
                M[i] = M[j] = true
                φ += ρij[i,j]
            end
        end
    end

    θ = 0.0
    variaveis = []
    for i in 1:N
        if M[i]
            push!(variaveis,i)
        end
        if u_opt[i] > 0.5
            θ += ρ0i[i]
        end
    end

    return Solution(0, obj, variaveis, θ, φ, 0.0)
end

function executeRegression(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    lin_reg = fit!(LinearRegression(), X_train, y_train)
    y_predict = predict(lin_reg, X_test)
    r2 = r2_score(y_test, y_predict)
    return r2
end

function executeNair(dataset)
    
    # Load data
    X, y, ρij, ρ0i = loadData(dataset)

    # Total of variables
    n = size(X,2)

    # Array to store Pareto front solutions
    solutionID = 1
    solutions = Solution[]
    solutionsTested = []

    ###########################
    # Get initial SS0 and SSF #
    ###########################

    #######################################
    # SS0 - variable with higher ρ0i value
    bestvariable = sortperm(ρ0i, rev = true)[1]
    θq = ρ0i[bestvariable]
    φq = 0
    
    # r2 for SS0
    X_new = reshape(X[:,bestvariable], length(X[:,bestvariable]), 1)
    r2_SS0 = executeRegression(X_new,y)

    ######################
    # SSF - all variables
    θr = sum(ρ0i)
    φr = 0.0
    for i in 1:n-1
        for j in i+1:n
            φr += ρij[i,j]
        end
    end

    # r2 for SSF
    r2_SSF = executeRegression(X,y)
    
    # αl for the first iteration
    αl = (φq - φr)/(θq + φq - θr - φr)

    # Objective value for SS0 and SSF
    zα = (αl * θq) - ( (1-αl) * φq )

    # Store solutions from pareto frontier SSO and SSF
    push!(solutions, Solution(solutionID,zα, [bestvariable], θq, φq, r2_SS0))
    solutionID += 1
    push!(solutions, Solution(solutionID, zα, collect(1:n), θr, φr, r2_SSF))
    solutionID += 1
    push!(solutionsTested,(solutions[1].id,solutions[2].id))

    #######################################
    # Solve the ILP with the calculated αl
    solution = executeVariableSelectModel(ρij, ρ0i, αl)

    if solution.obj > zα
        # Execute r2 for the new point
        X_new = X[:,solution.variables]
        solution.r2 = executeRegression(X_new,y)
        solution.id = solutionID 
        solutionID += 1
        insert!(solutions, 2, solution)
    end

    if size(solutions,1) == 3
        
        pointinserted = true

        while pointinserted == true
            
            pointinserted = false

            i = 1
            totalpontos = size(solutions,1)

            while i < totalpontos

                solution1 = solutions[i]
                solution2 = solutions[i+1]

                if (solution1.id, solution2.id) ∉ solutionsTested

                    # Calculate αl
                    αl = (solution1.φ - solution2.φ)/(solution1.θ + solution1.φ - solution2.θ - solution2.φ)

                    # Calculate zα for SS0
                    zα = (αl * solution1.θ) - ( (1-αl) * solution1.φ )

                    # Solve the ILP with the calculated αl
                    solutionNEW = executeVariableSelectModel(ρij, ρ0i, αl)

                    #if solutionNEW.obj > zα
                    if solutionNEW.obj - zα > 0.0001
                        # Execute r2 for the new point
                        X_new = X[:,solutionNEW.variables]
                        solutionNEW.r2 = executeRegression(X_new,y)
                        solutionNEW.id = solutionID
                        solutionID += 1
                        insert!(solutions,i+1,solutionNEW)
                        pointinserted = true
                        totalpontos += 1

                        push!(solutionsTested,(solution1.id, solution2.id))
                    end

                end

                i += 1 
            end

        end

    end

    # Plot Pareto Front
    XX = []
    XY = []
    for sol in solutions
        push!(XX,sol.φ)
        push!(XY,sol.θ)
    end
    py"printParetoFront"(XX,XY)
    display(gcf())
    savefig("results/paretoFront.pdf")

    py"printR2"(collect(1:size(solutions,1)), [sol.r2 for sol in solutions])
    display(gcf())
    savefig("results/r2.pdf")

    df = DataFrame(
        Point = collect(1:size(solutions,1)),
        Obj = [sol.obj for sol in solutions],
        Variables = [Int.(sol.variables) for sol in solutions],
        θ = [sol.θ for sol in solutions],
        φ = [sol.φ for sol in solutions],
        r2 = [sol.r2 for sol in solutions]
    )  

    CSV.write("results/resultado.csv",df)

    print(df)

    return solutions
end

#Random.seed!(0)
solutions = executeNair("datasets/2.txt");