<h1 id="mmpkan-dta">MMPKAN-DTA: A Multimodal Framework for Drug-Target Affinity Prediction Using Pretrained Models and Kolmogorov-Arnold Networks</h1>

<p align="center">
  <!-- Adjust width as needed (e.g., 800–1100) -->
  <img src="images/MMPKAN-DTA.jpg" alt="Model Architecture" width="900">
</p>
<p align="center"><em>Figure 1. MMPKAN-DTA model architecture.</em></p>

<p>This repository hosts <strong>MMPKAN-DTA</strong>, a deep learning framework for predicting drug–target affinities in drug discovery. Implemented in Python, the model leverages <strong>pretrained molecular and protein models</strong> alongside <strong>Kolmogorov–Arnold Networks (KAN)</strong> to achieve robust and reliable affinity predictions.</p>

---
<h2></h2>
<h2>📂 Repository Contents</h2>
<ul>
  <li><strong>📊 Data</strong>– Training dataset files. </li>
  <li><strong>💻 Code</strong>– Source code and scripts for the MMPKAN-DTA framework. </li>
</ul> 
---
<h2>🔀 Split Dataset for Both Warm- and Cold-start Scenarios</h2>
<pre><code>python code/cold_split.py</code></pre>

**Instructions:**  

- **Dataset Selection**  
  Set the dataset via the `dataset_name` parameter:  
  `davis`, `kiba`, or `metz`.  

- **Random Seed**  
  Use the `SEED` parameter to ensure reproducibility.  

- **Output**  
  The script generates **training**, **validation**, and **test** sets for:  
  - **warm**  
  - **novel-drug**  
  - **novel-prot**  
  - **novel-pair**  

> ⚠️ Ensure the dataset files are correctly placed in the **Data** folder before executing the script.

---
<h2>🗺 Generation of Pre-trained Embeddings</h2>
<pre><code>python pretrained/chemberta_pretraiend.py</code></pre>
<pre><code>python pretrained/esmC_pretraiend.py</code></pre>
<pre><code>python pretrained/esm2_map.py</code></pre>
<h2>▶️ Run the Code</h2>

Instructions for training, evaluation, and inference will be added here.
<h3>Train the Model</h3>
<pre><code>python code/train.py</code></pre>

<h3>Prediction</h3>
<pre><code>python code/test.py</code></pre>

---
<h2>📋 Requirements</h2>
<ul>
  <li>Python 3.9.21</li>
  <li>numpy==2.0.2</li>
  <li>pandas==2.2.3</li>
  <li>torch==2.6.0</li>
  <li>mamba-ssm==2.3.1</li>
  <li>causal-conv1d>=1.4.0</li>
  <li>rdkit==2024.3.2</li>
  <li>fair-esm==2.0.0</li>
</ul>

---
<h2>📖 Citation</h2>

<p>If you use this code or related methods in your research, please cite the <strong>MMPKAN-DTA</strong> paper:</p>  
<p><strong>(Citation details will be provided here once available)</strong></p>

<h2>📧 Contact</h2>
<p>For inquiries regarding this repository, please contact:
<strong>Muhammad Habibulla Alamin</strong> (Email: 
  <a href="mailto:habibulla.stat@gmail.com">habibulla.stat@gmail.com</a>)
</p>
