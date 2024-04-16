```plantuml
left to right direction
node "Training a WSI DL model" as train
node "Overfit plugin" as plugin {
    portin "DL model" as dlmodel
    portin embedding as emb
    portin input as input
    portin labels as labels

    node Adversarial
    node UMAP
    node Fairness as fairness
    
    /'
    portout umap
    portout adversarial
    portout fairness
    '/
}

train --> dlmodel
train --> emb
train --> input
train --> labels

dlmodel -> Adversarial
labels -> Adversarial
input -> Adversarial

emb -> UMAP

dlmodel -> fairness
labels -> fairness

node "Analysis ..." as analysis

plugin --> analysis
```
