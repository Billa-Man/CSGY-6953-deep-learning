# CSGY-6953-deep-learning-project-2

## Leaderboard Score
**Public:** 0.85500 <br>
**Private:** 0.84125

## Team Members

1. Sohith Bandari
   * NYU netID: sb10225
   * NYU email: sb10225@nyu.edu
   * kaggle ID: sohithbandari

2. Kushagra Yadav 
   * NYU netID: ky2684
   * NYU email: ky2684@nyu.edu
   * kaggle ID: kushagrayadv
  
3. Nishanth G Palaniswami
   * NYU netID: ng3124
   * NYU email: ng3124@nyu.edu
   * kaggle ID: nishanthgpalaniswami
  
## Prerequisites

- Python 3.12

## Installation Steps
First, clone the repository
```
git clone https://github.com/Billa-Man/CSGY-6953-deep-learning.git
cd <project-directory>/project2
```

### 1. Virtual Environment Setup
Create and activate a Python virtual environment:
```
# Create virtual environment
python3 -m venv virtual-env

# Activate virtual environment
# For Unix/macOS
source virtual-env/bin/activate

# For Windows
# virtual-env\Scripts\activate
```

### 2. Dependencies Installation
Install all required packages:
```
pip install datasets evaluate peft trl bitsandbytes wandb -q
pip install nvidia-ml-py3 -q
```

## Inference
Simply run the cells in the following notebook in your project directory after activating the environment:
```
final_notebook.ipynb
```
*Note* - The training and validation accuracy and loss curves are under the ``/screenshots`` folder.