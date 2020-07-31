# Algoritmos de Clasificación en Julia

findaccuracy(predictedvals,groundtruthvals) = sum(predictedvals.==groundtruthvals)/length(groundtruthvals)

#Configurando nuestro gestor de paquetes
using Pkg;

#Importando los paquetes necesarios
Pkg.add("GLMNet")
Pkg.add("RDatasets")
Pkg.add("MLBase")
Pkg.add("Plots")
Pkg.add("DecisionTree")
Pkg.add("Distances")
Pkg.add("NearestNeighbors")
Pkg.add("Random")
Pkg.add("LinearAlgebra")
Pkg.add("DataStructures")
Pkg.add("LIBSVM")


# Estableciendo el uso de los paquetes importados
using GLMNet
using RDatasets
using MLBase
using Plots
using DecisionTree
using Distances
using NearestNeighbors
using Random
using LinearAlgebra
using DataStructures
using LIBSVM


# Capturando los datos de prueba

iris = dataset("datasets", "iris")
first(iris, 5)

# Separando en la variable X la Matrix del conjunto de datos separados de la variable objetivo
X = Matrix(iris[:,1:4])
#Capturando las etiquetas (variable objetivo que nos interesa clasificar)
irislabels = iris[:,5]

#Convirtiendo en la variable categórica a variable numérica
irislabelsmap = labelmap(irislabels)
# asignando a la variable "y" la conversión
y = labelencode(irislabelsmap, irislabels)

#En la clasificación, a menudo queremos utilizar algunos de los datos 
#para ajustar un modelo, y el resto de los datos para validar 
# (comúnmente conocidos como datos de entrenamiento y pruebas). 
#Vamos a tener estos datos listos ahora para que podamos usarlos fácilmente 
#en el resto de este portátil.

function perclass_splits(y,at)
    uids = unique(y)
    keepids = []
    for ui in uids
        curids = findall(y.==ui)
        rowids = randsubseq(curids, at) 
        push!(keepids,rowids...)
    end
    return keepids
end

trainids = perclass_splits(y,0.7)
testids = setdiff(1:length(y),trainids)

# Necesitaremos una función más, y esa es la función que asignará 
# clases basadas en los valores pronosticados cuando los valores predichos 
# son continuos.

assign_class(predictedvalue) = argmin(abs.(predictedvalue .- [1,2,3]))

#Metodo 1 : Lasso

path = glmnet(X[trainids,:], y[trainids])
cv = glmnetcv(X[trainids,:], y[trainids])

# Escogiendo el mejor landa para predecir con este
path = glmnet(X[trainids,:], y[trainids])
cv = glmnetcv(X[trainids,:], y[trainids])
mylambda = path.lambda[argmin(cv.meanloss)]

path = glmnet(X[trainids,:], y[trainids],lambda=[mylambda]);

q = X[testids,:];
predictions_lasso = GLMNet.predict(path,q)

# Metodo 2 : Ridge

# choose the best lambda to predict with.
path = glmnet(X[trainids,:], y[trainids],alpha=0);
cv = glmnetcv(X[trainids,:], y[trainids],alpha=0)
mylambda = path.lambda[argmin(cv.meanloss)]
path = glmnet(X[trainids,:], y[trainids],alpha=0,lambda=[mylambda]);
q = X[testids,:];
predictions_ridge = GLMNet.predict(path,q)
predictions_ridge = assign_class.(predictions_ridge)
findaccuracy(predictions_ridge,y[testids])


#Metodo 3 : Elastic Net

# Usaremos la misma función pero estableceremos alfa en 0.5 (es la combinación de LASSO y RIDGE). Usaremos la misma función pero estableceremos alfa en 0.5 (es la combinación de LASSO y RIDGE).
# Elige la mejor lambda para predecir.
path = glmnet(X[trainids,:], y[trainids],alpha=0.5);
cv = glmnetcv(X[trainids,:], y[trainids],alpha=0.5)
mylambda = path.lambda[argmin(cv.meanloss)]
path = glmnet(X[trainids,:], y[trainids],alpha=0.5,lambda=[mylambda]);
q = X[testids,:];
predictions_EN = GLMNet.predict(path,q)
predictions_EN = assign_class.(predictions_EN)
findaccuracy(predictions_EN,y[testids])


# Metodo 4 : Árboles de Decisión

model = DecisionTreeClassifier(max_depth=2)
DecisionTree.fit!(model, X[trainids,:], y[trainids])

q = X[testids,:];
predictions_DT = DecisionTree.predict(model, q)
findaccuracy(predictions_DT,y[testids])

print_tree(model, 10)


using ScikitLearn.CrossValidation: cross_val_score
accuracy = cross_val_score(model, X[trainids,:], y[trainids], cv=3)

# Metodo 5 : Bosques Aleatorios

model = RandomForestClassifier(n_trees=20)
DecisionTree.fit!(model, X[trainids,:], y[trainids])

q = X[testids,:];
predictions_RF = DecisionTree.predict(model, q)
findaccuracy(predictions_RF,y[testids])

# Metodo 6 : Nearest Neighbor method

Xtrain = X[trainids,:]
ytrain = y[trainids]
kdtree = KDTree(Xtrain')

queries = X[testids,:]

idxs, dists = knn(kdtree, queries', 5, true)

c = ytrain[hcat(idxs...)]
possible_labels = map(i->counter(c[:,i]),1:size(c,2))
predictions_NN = map(i->parse(Int,string(argmax(DataFrame(possible_labels[i])[1,:]))),1:size(c,2))
findaccuracy(predictions_NN,y[testids])

#Metodo 7: Support Vector Machines

Xtrain = X[trainids,:]
ytrain = y[trainids]

model = svmtrain(Xtrain', ytrain)

predictions_SVM, decision_values = svmpredict(model, X[testids,:]')
findaccuracy(predictions_SVM,y[testids])

overall_accuracies = zeros(7)
methods = ["lasso","ridge","EN", "DT", "RF","kNN", "SVM"]
ytest = y[testids]
overall_accuracies[1] = findaccuracy(predictions_lasso,ytest)
overall_accuracies[2] = findaccuracy(predictions_ridge,ytest)
overall_accuracies[3] = findaccuracy(predictions_EN,ytest)
overall_accuracies[4] = findaccuracy(predictions_DT,ytest)
overall_accuracies[5] = findaccuracy(predictions_RF,ytest)
overall_accuracies[6] = findaccuracy(predictions_NN,ytest)
overall_accuracies[7] = findaccuracy(predictions_SVM,ytest)
hcat(methods, overall_accuracies)


