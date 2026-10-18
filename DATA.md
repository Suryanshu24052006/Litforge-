# Where the text comes from

LitForge uses five books by **Jane Austen**, downloaded from GITenberg’s copies of Project Gutenberg editions. The Austen texts are public domain in the United States. The downloaded raw files retain Project Gutenberg’s distribution terms. Raw book files are not included in this repository.

## Why five books?

| Job | Book |
|---|---|
| Learn patterns (training) | [Pride and Prejudice](https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/81db45c9c48c592f0b77f01fc59e677ad0a5634e/1342-0.txt) |
| Learn patterns (training) | [Sense and Sensibility](https://raw.githubusercontent.com/GITenberg/Sense-and-Sensibility_161/1cd7ffed1c21536f6ee2fea4d68c96e2c12ce14c/161.txt) |
| Learn patterns (training) | [Emma](https://raw.githubusercontent.com/GITenberg/Emma_158/9943f6422b9246e7ad37625d6bd014b035fbfdea/158-0.txt) |
| Choose a saved model (validation) | [Persuasion](https://raw.githubusercontent.com/GITenberg/Persuasion_105/d408e95fc4098953e8dd8eb7cac6023802cb54f4/105.txt) |
| Measure the final result (testing) | [Northanger Abbey](https://raw.githubusercontent.com/GITenberg/Northanger-Abbey_121/c1b554e6d17c67d26b4fbd4bcc92d7eb4f27b2e7/121-0.txt) |

Keeping these jobs separate helps check whether a model can predict a book it did not train on. The final test is scored after all planned training runs finish.

## How the books are prepared

1. Download the exact recorded version of each book.
2. Check its file fingerprint, called a checksum, to catch changed contents.
3. Remove the Gutenberg header and footer and make line endings consistent.
4. Save the cleaned text and record another checksum.

Training passages stay inside one book. A passage never starts in one book and ends in another. Evaluation also starts with fresh context at each book and each fixed-size window.

The preparation step checks for long repeated paragraphs across the splits. It found **zero exact matches of at least 100 characters after whitespace normalization**. Short phrases, paraphrases, and shared author style can still overlap. Preparation reads all books for cleaning and this audit; the training loader itself does not open the test book.

## Check the source details

The [dataset record](artifacts/study/dataset.json) stores exact source versions, download URLs, raw and cleaned checksums, byte counts, and the overlap check. The implementation is in `litforge/books.py`. If a download changes or the expected book markers are missing, preparation stops with an error.

These checks make the files repeatable; they do not show that the data represents other authors, genres, or languages. Historical language and stereotypes in the books can appear in generated text.

## The older Shakespeare experiment

The original experiment used [Tiny Shakespeare from char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare). It split one text file into consecutive training, validation, and test sections, which gives weaker separation than using different books. Its checksum and preparation code are in `litforge/data.py`; the original commands remain available.

Those scores are per character. The current study uses different books and scores per UTF-8 byte, so the two experiments’ scores cannot be compared directly.
