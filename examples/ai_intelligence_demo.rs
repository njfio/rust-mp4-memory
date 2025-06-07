use rust_mem_vid::{MemvidEncoder, Config};
use rust_mem_vid::knowledge_graph::{KnowledgeGraphBuilder, ConceptNode};
use rust_mem_vid::concept_extractors::{NamedEntityExtractor, KeywordExtractor, TechnicalConceptExtractor};
use rust_mem_vid::relationship_analyzers::{CooccurrenceAnalyzer, SemanticSimilarityAnalyzer, HierarchicalAnalyzer};
use rust_mem_vid::content_synthesis::{ContentSynthesizer, SynthesisType, TemplateSynthesisStrategy};
use rust_mem_vid::analytics_dashboard::{AnalyticsDashboard, RawAnalyticsData};
use std::fs;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🧠 AI Intelligence Features Demo");
    println!("================================");
    println!("This demo showcases Phase 1 AI Intelligence features:");
    println!("• 🕸️  Knowledge Graph Generation");
    println!("• 🤖 Intelligent Content Synthesis");
    println!("• 📊 Advanced Analytics Dashboard");
    println!();
    
    // Create test content with rich knowledge
    let temp_dir = create_ai_test_content().await?;
    let demo_path = temp_dir.path();
    
    println!("📁 Created AI test content at: {}", demo_path.display());
    
    // Demo 1: Create memory with rich content
    println!("\n🎬 Demo 1: Creating memory with AI-rich content");
    let (video_path, index_path) = create_ai_memory(demo_path).await?;
    
    // Demo 2: Knowledge Graph Generation
    println!("\n🕸️  Demo 2: Knowledge Graph Generation");
    demo_knowledge_graph_generation(&video_path, &index_path).await?;
    
    // Demo 3: Intelligent Content Synthesis
    println!("\n🤖 Demo 3: Intelligent Content Synthesis");
    demo_content_synthesis(&video_path, &index_path).await?;
    
    // Demo 4: Advanced Analytics Dashboard
    println!("\n📊 Demo 4: Advanced Analytics Dashboard");
    demo_analytics_dashboard(&video_path, &index_path).await?;
    
    println!("\n✅ All AI Intelligence demos completed!");
    println!("🎉 Revolutionary capabilities demonstrated:");
    println!("   • Automatic concept extraction and relationship mapping");
    println!("   • Intelligent content synthesis and insight generation");
    println!("   • Advanced analytics with visual knowledge tracking");
    println!("   • AI-powered knowledge evolution analysis");
    
    Ok(())
}

async fn create_ai_test_content() -> anyhow::Result<TempDir> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create AI and Machine Learning content
    fs::write(base_path.join("ai_overview.md"), r#"
# Artificial Intelligence Overview

Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. AI encompasses various subfields including machine learning, natural language processing, computer vision, and robotics.

## Key Concepts

### Machine Learning
Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. Popular algorithms include:

- **Neural Networks**: Inspired by biological neural networks, these are computing systems designed to recognize patterns
- **Deep Learning**: A subset of machine learning using neural networks with multiple layers
- **Supervised Learning**: Learning with labeled training data
- **Unsupervised Learning**: Finding patterns in data without labeled examples
- **Reinforcement Learning**: Learning through interaction with an environment

### Natural Language Processing (NLP)
NLP enables computers to understand, interpret, and generate human language. Key applications include:
- Sentiment analysis
- Language translation
- Chatbots and virtual assistants
- Text summarization

### Computer Vision
Computer vision enables machines to interpret and understand visual information from the world. Applications include:
- Image recognition and classification
- Object detection and tracking
- Facial recognition
- Medical image analysis

## Major Players

### Technology Companies
- **Google**: Developed TensorFlow, BERT, and various AI services
- **Microsoft**: Created Azure AI services and invested heavily in OpenAI
- **Amazon**: Provides AWS AI services and developed Alexa
- **OpenAI**: Created GPT models and ChatGPT
- **Meta**: Developed PyTorch and various AI research initiatives

### Research Institutions
- **MIT**: Leading AI research with the Computer Science and Artificial Intelligence Laboratory (CSAIL)
- **Stanford University**: Home to the Stanford AI Lab and Human-Centered AI Institute
- **Carnegie Mellon University**: Renowned for robotics and machine learning research
- **University of Toronto**: Pioneered deep learning research

## Current Trends

### Large Language Models (LLMs)
Recent breakthroughs in large language models have revolutionized AI:
- GPT-3 and GPT-4 by OpenAI
- BERT and LaMDA by Google
- Claude by Anthropic
- LLaMA by Meta

### Generative AI
AI systems capable of creating new content:
- Text generation (ChatGPT, Claude)
- Image generation (DALL-E, Midjourney, Stable Diffusion)
- Code generation (GitHub Copilot, CodeT5)
- Music and video generation

### AI Ethics and Safety
Growing focus on responsible AI development:
- Bias detection and mitigation
- Explainable AI (XAI)
- AI alignment and safety research
- Regulatory frameworks and governance
"#)?;

    fs::write(base_path.join("ml_algorithms.md"), r#"
# Machine Learning Algorithms

This document provides an overview of fundamental machine learning algorithms and their applications.

## Supervised Learning Algorithms

### Linear Regression
Linear regression is used for predicting continuous values by finding the best linear relationship between input features and target values.

**Applications:**
- House price prediction
- Sales forecasting
- Risk assessment

**Advantages:**
- Simple to understand and implement
- Fast training and prediction
- No hyperparameter tuning required

### Decision Trees
Decision trees create a model that predicts target values by learning simple decision rules inferred from data features.

**Applications:**
- Medical diagnosis
- Credit approval
- Feature selection

**Advantages:**
- Easy to interpret and visualize
- Handles both numerical and categorical data
- Requires little data preparation

### Random Forest
Random Forest builds multiple decision trees and merges them together to get more accurate and stable predictions.

**Applications:**
- Image classification
- Bioinformatics
- Stock market analysis

**Advantages:**
- Reduces overfitting compared to decision trees
- Handles missing values well
- Provides feature importance rankings

### Support Vector Machines (SVM)
SVM finds the optimal boundary (hyperplane) that separates different classes with maximum margin.

**Applications:**
- Text classification
- Image recognition
- Gene classification

**Advantages:**
- Effective in high-dimensional spaces
- Memory efficient
- Versatile with different kernel functions

## Unsupervised Learning Algorithms

### K-Means Clustering
K-Means partitions data into k clusters where each observation belongs to the cluster with the nearest mean.

**Applications:**
- Customer segmentation
- Image segmentation
- Data compression

**Process:**
1. Choose number of clusters (k)
2. Initialize cluster centroids randomly
3. Assign each point to nearest centroid
4. Update centroids based on assigned points
5. Repeat until convergence

### Hierarchical Clustering
Hierarchical clustering creates a tree of clusters by iteratively merging or splitting clusters.

**Types:**
- **Agglomerative**: Bottom-up approach, starts with individual points
- **Divisive**: Top-down approach, starts with all points in one cluster

**Applications:**
- Phylogenetic analysis
- Social network analysis
- Market research

### Principal Component Analysis (PCA)
PCA reduces dimensionality by finding the principal components that explain the most variance in the data.

**Applications:**
- Dimensionality reduction
- Data visualization
- Feature extraction
- Noise reduction

**Benefits:**
- Reduces computational complexity
- Eliminates multicollinearity
- Helps visualize high-dimensional data

## Deep Learning Algorithms

### Convolutional Neural Networks (CNNs)
CNNs are designed to process grid-like data such as images using convolutional layers.

**Architecture Components:**
- Convolutional layers
- Pooling layers
- Fully connected layers
- Activation functions (ReLU, Sigmoid, Tanh)

**Applications:**
- Image classification
- Object detection
- Medical image analysis
- Autonomous vehicles

### Recurrent Neural Networks (RNNs)
RNNs are designed to work with sequential data by maintaining hidden states that capture information from previous time steps.

**Variants:**
- **LSTM**: Long Short-Term Memory networks
- **GRU**: Gated Recurrent Units
- **Bidirectional RNNs**: Process sequences in both directions

**Applications:**
- Natural language processing
- Speech recognition
- Time series prediction
- Machine translation

### Transformer Architecture
Transformers use self-attention mechanisms to process sequential data more efficiently than RNNs.

**Key Components:**
- Multi-head attention
- Position encoding
- Feed-forward networks
- Layer normalization

**Applications:**
- Language models (GPT, BERT)
- Machine translation
- Text summarization
- Code generation

## Reinforcement Learning

### Q-Learning
Q-Learning learns the quality of actions, telling an agent what action to take under what circumstances.

**Components:**
- **Agent**: The learner or decision maker
- **Environment**: The world the agent interacts with
- **Actions**: What the agent can do
- **Rewards**: Feedback from the environment
- **Policy**: Strategy for choosing actions

**Applications:**
- Game playing (Chess, Go, Atari games)
- Robotics control
- Trading algorithms
- Resource allocation

### Deep Q-Networks (DQN)
DQN combines Q-Learning with deep neural networks to handle high-dimensional state spaces.

**Innovations:**
- Experience replay
- Target networks
- Double DQN
- Dueling DQN

**Famous Applications:**
- AlphaGo and AlphaZero
- Atari game playing
- Autonomous driving
- Resource management

## Algorithm Selection Guidelines

### Factors to Consider
1. **Problem Type**: Classification, regression, clustering, or reinforcement learning
2. **Data Size**: Small datasets may benefit from simpler algorithms
3. **Interpretability**: Some applications require explainable models
4. **Training Time**: Real-time applications need fast training algorithms
5. **Accuracy Requirements**: High-stakes decisions may require ensemble methods

### Performance Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Regression**: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared
- **Clustering**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- **Reinforcement Learning**: Cumulative Reward, Episode Length, Success Rate

## Future Directions

### Emerging Trends
- **Few-shot Learning**: Learning from limited examples
- **Meta-Learning**: Learning to learn new tasks quickly
- **Federated Learning**: Training models across decentralized data
- **Neuromorphic Computing**: Brain-inspired computing architectures
- **Quantum Machine Learning**: Leveraging quantum computing for ML

### Challenges
- **Data Privacy**: Protecting sensitive information in ML models
- **Model Interpretability**: Understanding complex model decisions
- **Robustness**: Ensuring models work reliably in real-world conditions
- **Computational Efficiency**: Reducing energy consumption and training time
- **Ethical AI**: Addressing bias, fairness, and societal impact
"#)?;

    fs::write(base_path.join("ai_research_papers.md"), r#"
# Influential AI Research Papers

This document catalogs groundbreaking research papers that have shaped the field of artificial intelligence.

## Foundational Papers

### "Computing Machinery and Intelligence" (1950)
**Author:** Alan Turing  
**Significance:** Introduced the Turing Test as a criterion for machine intelligence.

**Key Contributions:**
- Proposed the imitation game (Turing Test)
- Discussed the possibility of thinking machines
- Laid philosophical foundations for AI

### "A Logical Calculus of Ideas Immanent in Nervous Activity" (1943)
**Authors:** Warren McCulloch and Walter Pitts  
**Significance:** Introduced the first mathematical model of neural networks.

**Key Contributions:**
- Mathematical model of neurons
- Foundation for artificial neural networks
- Demonstrated computational capabilities of neural networks

## Machine Learning Breakthroughs

### "Learning Representations by Back-propagating Errors" (1986)
**Authors:** David Rumelhart, Geoffrey Hinton, and Ronald Williams  
**Significance:** Popularized the backpropagation algorithm for training neural networks.

**Key Contributions:**
- Efficient algorithm for training multi-layer neural networks
- Enabled practical deep learning applications
- Solved the credit assignment problem

### "Support-Vector Networks" (1995)
**Authors:** Corinna Cortes and Vladimir Vapnik  
**Significance:** Introduced Support Vector Machines (SVMs).

**Key Contributions:**
- Margin-based classification approach
- Kernel trick for non-linear classification
- Strong theoretical foundations

### "Random Forests" (2001)
**Author:** Leo Breiman  
**Significance:** Introduced the Random Forest ensemble method.

**Key Contributions:**
- Ensemble learning approach
- Reduced overfitting compared to individual decision trees
- Built-in feature importance measures

## Deep Learning Revolution

### "ImageNet Classification with Deep Convolutional Neural Networks" (2012)
**Authors:** Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton  
**Significance:** Demonstrated the power of deep CNNs for image classification.

**Key Contributions:**
- AlexNet architecture
- Significant improvement on ImageNet challenge
- Sparked the deep learning revolution

### "Deep Residual Learning for Image Recognition" (2015)
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun  
**Significance:** Introduced ResNet architecture with skip connections.

**Key Contributions:**
- Residual connections to enable very deep networks
- Solved vanishing gradient problem
- Achieved human-level performance on ImageNet

### "Generative Adversarial Networks" (2014)
**Authors:** Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, et al.  
**Significance:** Introduced GANs for generative modeling.

**Key Contributions:**
- Adversarial training framework
- High-quality image generation
- Foundation for many generative AI applications

## Natural Language Processing

### "Attention Is All You Need" (2017)
**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.  
**Significance:** Introduced the Transformer architecture.

**Key Contributions:**
- Self-attention mechanism
- Parallelizable architecture
- Foundation for modern language models

### "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
**Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova  
**Significance:** Demonstrated the power of bidirectional pre-training.

**Key Contributions:**
- Bidirectional context understanding
- Transfer learning for NLP
- State-of-the-art results on multiple NLP tasks

### "Language Models are Few-Shot Learners" (2020)
**Authors:** Tom Brown, Benjamin Mann, Nick Ryder, et al.  
**Significance:** Introduced GPT-3 and demonstrated few-shot learning capabilities.

**Key Contributions:**
- 175 billion parameter language model
- In-context learning without fine-tuning
- Demonstrated emergent abilities at scale

## Reinforcement Learning

### "Playing Atari with Deep Reinforcement Learning" (2013)
**Authors:** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al.  
**Significance:** Introduced Deep Q-Networks (DQN).

**Key Contributions:**
- Combined deep learning with reinforcement learning
- End-to-end learning from raw pixels
- Achieved human-level performance on Atari games

### "Mastering the Game of Go with Deep Neural Networks and Tree Search" (2016)
**Authors:** David Silver, Aja Huang, Chris Maddison, et al.  
**Significance:** Introduced AlphaGo, which defeated world champion Go players.

**Key Contributions:**
- Monte Carlo Tree Search with neural networks
- Policy and value networks
- Demonstrated AI superiority in complex strategic games

### "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (2017)
**Authors:** David Silver, Thomas Hubert, Julian Schrittwieser, et al.  
**Significance:** Introduced AlphaZero, which learned games from scratch.

**Key Contributions:**
- Self-play learning without human knowledge
- General algorithm for multiple games
- Superhuman performance through pure reinforcement learning

## Computer Vision

### "You Only Look Once: Unified, Real-Time Object Detection" (2015)
**Authors:** Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi  
**Significance:** Introduced YOLO for real-time object detection.

**Key Contributions:**
- Single-shot object detection
- Real-time performance
- End-to-end trainable system

### "Mask R-CNN" (2017)
**Authors:** Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick  
**Significance:** Extended object detection to instance segmentation.

**Key Contributions:**
- Instance segmentation framework
- Multi-task learning approach
- High-quality segmentation masks

## AI Safety and Alignment

### "Concrete Problems in AI Safety" (2016)
**Authors:** Dario Amodei, Chris Olah, Jacob Steinhardt, et al.  
**Significance:** Outlined key challenges in AI safety research.

**Key Contributions:**
- Identified five practical safety problems
- Research agenda for AI safety
- Bridge between theoretical and practical safety concerns

### "Constitutional AI: Harmlessness from AI Feedback" (2022)
**Authors:** Yuntao Bai, Andy Jones, Kamal Ndousse, et al.  
**Significance:** Introduced methods for training helpful, harmless, and honest AI systems.

**Key Contributions:**
- Constitutional training approach
- AI feedback for alignment
- Scalable oversight methods

## Recent Developments

### "Training Language Models to Follow Instructions with Human Feedback" (2022)
**Authors:** Long Ouyang, Jeff Wu, Xu Jiang, et al.  
**Significance:** Introduced InstructGPT and RLHF (Reinforcement Learning from Human Feedback).

**Key Contributions:**
- Human feedback for model alignment
- Instruction-following capabilities
- Foundation for ChatGPT and similar systems

### "PaLM: Scaling Language Modeling with Pathways" (2022)
**Authors:** Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al.  
**Significance:** Demonstrated capabilities of 540B parameter language models.

**Key Contributions:**
- Breakthrough scaling of language models
- Emergent reasoning capabilities
- Multi-task learning at scale

## Impact and Future Directions

### Research Trends
- **Multimodal AI**: Combining vision, language, and other modalities
- **Foundation Models**: Large-scale pre-trained models for multiple tasks
- **AI Alignment**: Ensuring AI systems behave as intended
- **Efficient AI**: Reducing computational requirements and energy consumption
- **Interpretable AI**: Understanding how AI systems make decisions

### Societal Impact
- **Healthcare**: AI-assisted diagnosis and drug discovery
- **Education**: Personalized learning and intelligent tutoring systems
- **Climate**: AI for climate modeling and sustainable technology
- **Scientific Discovery**: AI-accelerated research and hypothesis generation
- **Creative Industries**: AI tools for art, music, and content creation

These papers represent milestones in AI development and continue to influence current research directions and applications.
"#)?;

    Ok(temp_dir)
}

async fn create_ai_memory(demo_path: &std::path::Path) -> anyhow::Result<(String, String)> {
    println!("   Creating AI-rich memory from content...");
    
    let config = Config::default();
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    // Add all AI content files
    encoder.add_directory(&demo_path.to_string_lossy()).await?;
    
    // Build video
    let video_path = demo_path.join("ai_memory.mp4");
    let index_path = demo_path.join("ai_memory");
    
    let stats = encoder.build_video(
        video_path.to_str().unwrap(),
        index_path.to_str().unwrap()
    ).await?;
    
    println!("     ✅ {} chunks encoded in {:.2}s", 
             stats.total_chunks, stats.encoding_time_seconds);
    
    Ok((
        video_path.to_str().unwrap().to_string(),
        format!("{}.metadata", index_path.to_str().unwrap())
    ))
}

async fn demo_knowledge_graph_generation(video_path: &str, index_path: &str) -> anyhow::Result<()> {
    println!("   Building knowledge graph from AI content...");
    
    let config = Config::default();
    let mut graph_builder = KnowledgeGraphBuilder::new(config)
        .with_embeddings().await?;
    
    // Add concept extractors
    graph_builder = graph_builder
        .add_concept_extractor(Box::new(NamedEntityExtractor::new()?))
        .add_concept_extractor(Box::new(KeywordExtractor::new()))
        .add_concept_extractor(Box::new(TechnicalConceptExtractor::new()?));
    
    // Add relationship analyzers
    graph_builder = graph_builder
        .add_relationship_analyzer(Box::new(CooccurrenceAnalyzer::new()?))
        .add_relationship_analyzer(Box::new(SemanticSimilarityAnalyzer::new()))
        .add_relationship_analyzer(Box::new(HierarchicalAnalyzer::new()?));
    
    // Build knowledge graph
    let memories = vec![(video_path.to_string(), index_path.to_string())];
    let knowledge_graph = graph_builder.build_from_memories(&memories).await?;
    
    println!("     🕸️  Knowledge Graph Generated:");
    println!("        • Concepts: {}", knowledge_graph.nodes.len());
    println!("        • Relationships: {}", knowledge_graph.relationships.len());
    println!("        • Communities: {}", knowledge_graph.communities.len());
    
    // Show some example concepts
    let mut concepts: Vec<&ConceptNode> = knowledge_graph.nodes.values().collect();
    concepts.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap());
    
    println!("     🔝 Top Concepts:");
    for concept in concepts.iter().take(5) {
        println!("        • {} (importance: {:.2}, type: {:?})", 
                 concept.name, concept.importance_score, concept.concept_type);
    }
    
    Ok(())
}

async fn demo_content_synthesis(video_path: &str, index_path: &str) -> anyhow::Result<()> {
    println!("   Generating intelligent content synthesis...");
    
    let config = Config::default();
    let synthesizer = ContentSynthesizer::new(config)
        .add_strategy(Box::new(TemplateSynthesisStrategy::new()));
    
    let memories = vec![(video_path.to_string(), index_path.to_string())];
    
    // Generate different types of synthesis
    println!("     🤖 Generating AI Insights:");
    
    // Summary synthesis
    let summary = synthesizer.generate_summary("machine learning algorithms", &memories).await?;
    println!("        📋 Summary (confidence: {:.1}%):", summary.confidence * 100.0);
    println!("           {}", &summary.content[..200.min(summary.content.len())]);
    if summary.content.len() > 200 { println!("           ..."); }
    
    // Insights synthesis
    let insights = synthesizer.extract_insights("deep learning trends", &memories).await?;
    println!("        💡 Insights (confidence: {:.1}%):", insights.confidence * 100.0);
    println!("           Key points: {}", insights.key_points.len());
    for (i, point) in insights.key_points.iter().take(3).enumerate() {
        println!("           {}. {} (importance: {:.2})", i + 1, point.point, point.importance);
    }
    
    // Knowledge gaps
    let gaps = synthesizer.identify_knowledge_gaps("quantum machine learning", &memories).await?;
    println!("        🔍 Knowledge Gaps (confidence: {:.1}%):", gaps.confidence * 100.0);
    println!("           {}", &gaps.content[..150.min(gaps.content.len())]);
    if gaps.content.len() > 150 { println!("           ..."); }
    
    Ok(())
}

async fn demo_analytics_dashboard(video_path: &str, index_path: &str) -> anyhow::Result<()> {
    println!("   Generating advanced analytics dashboard...");
    
    let config = Config::default();
    let dashboard = AnalyticsDashboard::new(config);
    
    // Create mock analytics data
    let raw_data = RawAnalyticsData {
        timelines: Vec::new(),
        snapshots: Vec::new(),
        diffs: Vec::new(),
        knowledge_graphs: Vec::new(),
        query_logs: Vec::new(),
        performance_metrics: Vec::new(),
    };
    
    // Generate dashboard
    let dashboard_output = dashboard.generate_dashboard(raw_data).await?;
    
    println!("     📊 Analytics Dashboard Generated:");
    println!("        • Visualizations: {}", dashboard_output.visualizations.len());
    println!("        • Insights: {}", dashboard_output.insights.len());
    println!("        • Recommendations: {}", dashboard_output.recommendations.len());
    
    // Show insights
    println!("     💡 Generated Insights:");
    for insight in &dashboard_output.insights {
        println!("        • {} (importance: {:.1}%)", insight.title, insight.importance * 100.0);
        println!("          {}", insight.description);
    }
    
    // Show recommendations
    println!("     🎯 Generated Recommendations:");
    for rec in &dashboard_output.recommendations {
        println!("        • {} [{}]", rec.title, rec.priority);
        println!("          {} (impact: {:.1}%)", rec.description, rec.estimated_impact * 100.0);
    }
    
    Ok(())
}
