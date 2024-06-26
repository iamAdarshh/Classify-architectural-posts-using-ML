# Architectural Posts Classification on Stack Overflow

## Overview

This repository contains an assignment project focused on classifying architectural posts from Stack Overflow using machine learning techniques. The data for this project is sourced from Stack Overflow, which is freely available and structured in a Microsoft SQL Server database with around 18 tables. The primary table of interest is `Posts`.

## Table of Contents

- [Data Source](#data-source)
- [Data Structure](#data-structure)
- [Types of Posts](#types-of-posts)
- [Classification Methodology](#classification-methodology)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Data Source

The data used in this project is sourced from Stack Overflow and can be accessed through the following means:

- **Query Interface**: You can retrieve specific information using the [Stack Exchange Data Explorer](https://data.stackexchange.com/stackoverflow/query/new).
- **Database XML Dump**: The complete dataset is available for download from [Archive.org](https://archive.org/details/stackexchange).
- **Stack Exchange API:** [Documentation](https://api.stackexchange.com/). Issues with throttles https://api.stackexchange.com/docs/throttle.

## Data Structure

The primary table of interest in this project is the `Posts` table. Key fields in this table include:

- `PostTypeId`: Indicates whether the post is a question (1) or an answer (2).
- `ParentId`: For answers, this field contains the ID of the corresponding question.

## Types of Posts

On Stack Overflow, architectural posts can be classified based on their purpose and type of solution. Each architectural post will have a specific purpose and solution type:

### Purpose

- **Explanatory**: Posts focused on explaining solutions.
- **Evaluation**: Posts evaluating one or more solutions.
- **Synthesis**: Posts concerned with finding or integrating multiple solutions.

### Type of Solution

- **Technology**: Solutions involving libraries or frameworks.
- **Conceptual**: Solutions involving patterns or tactics.
- **Technology Feature**: Solutions discussing specific functionalities.

## Classification Methodology

The methodology to classify architectural posts involves the following steps:

1. **Data Collection**: Gather posts from Stack Overflow using the provided query interface or database dump.
2. **Data Preprocessing**: Clean and preprocess the data to ensure it's suitable for training machine learning models.
3. **Feature Extraction**: Extract relevant features from the posts that can help in classification.
4. **Model Training**: Train machine learning models to classify the posts based on their purpose and solution type.
5. **Evaluation**: Evaluate the performance of the models using appropriate metrics.

Here's an enhanced version of the setup section for your repository's README file:

---

## Setup Instructions

Follow these steps to set up the project locally:

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/architectural-posts-classification.git
cd architectural-posts-classification
```

### 2. Install Dependencies

Ensure you have Python and pip installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

To access the necessary data, you'll need to create an API key from StackExchange and add it to the project:

1. Create an account on [StackExchange](https://api.stackexchange.com/).
2. Configure your application to obtain an API key.
3. Create a file named `apikey.txt` in the project root directory and paste your API key inside it.

### 4. Download Input Data

Download the required input data by running the appropriate script for your operating system:

- **For Windows**:
    ```bash
    download_data.bat
    ```
- **For macOS**:
    ```shell
    ./download_data.sh
    ```


---

This enhanced version includes clearer instructions, consistency in formatting, and an additional step to verify the setup, which helps ensure that users have correctly followed the setup process.
     
## Contributing

Contributions to this project are welcome. Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

By following the steps and guidelines provided in this README, you should be able to effectively work on the assignment to classify architectural posts from Stack Overflow using machine learning. Happy coding!