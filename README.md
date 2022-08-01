{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "README.md",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/LittleBigKeith/Computer-Vision/blob/master/README.md\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Toxic Comment Classifier  \n",
        "#####  Chenyang Xu, Hei Lok Keith Kong, Nicholas Bor , Shuangjian Li , Van Minh Tran  \n",
        "###   \n",
        "---------------\n",
        "## Introduction\n",
        "This project is based on the [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) on Kaggle. The goal is to develop models that Identify and classify toxic online comments.\n",
        "\n",
        "This code uses Python 3.9. Several modules may need to be imported including: nltk, numpy, pandas, spaCy, sckitlearn, imblearn, tensorflow, keras and gc.\n",
        "\n",
        "In particular, to install [spaCy](https://spacy.io/usage):\n",
        "\n",
        "    pip install -U pip setuptools wheel\n",
        "    pip install -U spacy\n",
        "    python -m spacy download en_core_web_sm\n",
        "\n",
        "To install [imblearn](https://pypi.org/project/imbalanced-learn/):\n",
        "\n",
        "    pip install -U imbalanced-learn\n",
        "   \n",
        "To install [tensorflow](https://www.tensorflow.org/install/pip):\n",
        "\n",
        "    python3 -m pip install tensorflow\n",
        "\n",
        "To install [keras](https://pypi.org/project/keras/):\n",
        "\n",
        "    pip install keras\n",
        "\n",
        "-------\n",
        "\n",
        "## Data\n",
        "The [train.csv](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip) file can be downloaded from Kaggle and is also in zip, extracted and uploaded to the notebook in order to process the data and pass it to the models.\n",
        "\n",
        "----------\n",
        "\n",
        "## Data pre-processing\n",
        "The  first part of this project uses regular expression and spaCy to perform data cleanup. \n",
        "\n",
        "`pandas` is used to load `train.csv` and sample 20% of the examples.   \n",
        "Regular expression is used to remove hyperlinks, punctuations and redundant characters and whitespace.   \n",
        "\n",
        "`spaCy` is used to remove stop words and digit-like characters.\n",
        "\n",
        "------------\n",
        "\n",
        "\n",
        "## Models\n",
        "We implemented the following models:\n",
        " 1. Multinomial NB\n",
        " 2. Logistic Regression\n",
        " 3. Bernoulli NB\n",
        " 4. kNN\n",
        " 5. Neural Network\n",
        " \n",
        "In each of the model, we trained and fit a model based on training data, tuned hyperparameters with validation data and evaluate metrics based on unseen test data. The metrics include accuracy, precision, recall and f1 score. Confusion matrices are plot to visualize the predictions versus the true labels."
      ],
      "metadata": {
        "id": "YJ41WufdcdEb"
      }
    }
  ]
}