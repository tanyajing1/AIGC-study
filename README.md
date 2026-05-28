# ✨Pytorch文件夹：
  上传了学习莫烦Pytorch训练课程的所有练习代码，完成了从零开始学习Pytorch以及神经网络基本结构。
# ✨mini_transformer文件夹：
  实现了用Pytorch从零实现一个极简的Transformer block，并使用一个toy dataset数据集训练模型来实现了一个英➡️中的翻译。
# ✨CLIP文件夹：
  实现了用HuggingFace加载CLIP模型，来实现一个简单的图文相似性代码。
# ✨CLIP_image-text_matching文件夹：
  跑一个图文匹配模型（CLIP），实现了用文找图的功能。


flowchart TB
    %% Styling Definitions
    classDef data fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1,rx:5px,ry:5px
    classDef model fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#4A148C,rx:10px,ry:10px
    classDef module fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20,rx:5px,ry:5px
    classDef attention fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#E65100,rx:5px,ry:5px
    classDef output fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#B71C1C,rx:10px,ry:10px

    %% Inputs
    subgraph Inputs [Inputs]
        direction LR
        I["🖼️ Source Image"]:::data
        SP["📝 Source Prompt"]:::data
        TP["✍️ Target Prompt"]:::data
    end

    %% Pre-processing
    subgraph ALE_Phase [ALE Pre-processing]
        SAM["Segment Anything Model (SAM)<br/>Cross-Attention Masking"]:::model
        Masks["Foreground & Background<br/>Spatial Masks"]:::data
    end

    %% Core Pipeline
    subgraph MMALE_Core [MMALE Core Pipeline (Modified U-Net)]
        LDM["Latent Diffusion Process"]:::model
        
        subgraph Masa_Block [MasaCtrl Integration]
            direction TB
            Trigger["Step / Layer Condition<br/>(masa_start_step, masa_start_layer)"]:::module
            
            subgraph Attn_Mech [Decoupled Self-Attention Mechanism]
                direction LR
                Q["Query (Q)<br/>Enables Non-Rigid<br/>Structural Transformation"]:::attention
                KV["Key (K) & Value (V)<br/>Preserves Original<br/>Visual Appearance"]:::attention
            end
        end

        subgraph Fusion_Block [Ablation Modules: Background Preservation]
            direction TB
            DBM["Dynamic Background Mask<br/>(Adaptive Mask Generation)"]:::module
            DBF["Delayed Background Fusion<br/>(Temporal Feature Blending)"]:::module
        end
    end

    %% Final Output
    Out["✨ Final Edited Image<br/>(High Quality & Consistency)"]:::output

    %% Data Flow
    I --> SAM
    SP --> SAM
    TP --> SAM
    SAM --> Masks

    I -.-> LDM
    SP -.-> LDM
    TP -.-> LDM

    LDM ==> Trigger
    Trigger ==> Attn_Mech
    
    Attn_Mech ==> Fusion_Block
    Masks --> DBM
    
    Fusion_Block ==> Out
