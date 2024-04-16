```plantuml
left to right direction
node "Model developer" as developer {
    node "Training data" as train
    node "Annotations" as anno {
        node Labels as labels
        node "Sensitive variables" as sens
    }
}
node MONAI as monai {
    node "Imageloader" as imageloader
    node "PyTorch" as pytorch
    node "Distributed trainer"
}

node "Model SSL training" as trainer {
    node "Conditional Sampler" as sampler
}
node "Models" as models
node "Overfit exploration plugin" as overfit {
    node "Metric #1" as metric1
    node "Metric #2" as metric2
    node "Metric #3" as metric3
}
map "Analysis output" as analysis {
    Metric #1 => α, β, γ
    Metric #2 => α, β, γ
    Metric #3 => α, β, γ

}

train ---> trainer
sens --> sampler
' sens -> overfit
trainer <.down... monai
trainer --> models
models --> overfit 
overfit <.up... monai
overfit --> analysis
```
