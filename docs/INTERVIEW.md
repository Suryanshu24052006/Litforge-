# Explain LitForge in an interview

## Start with this short explanation

> I built a small project to understand how AI generates text. You give it an opening, and it tries to continue the passage. I compared two models trained on the same books: a Transformer and a GRU. I also tested a simple method based on counting short text patterns. The simple method actually won at this training budget. I built an app so you can try the models and explore their predictions.

Use your own words. Be ready to run the project and explain the code you discuss.

## How does it learn?

Imagine training on the word `HELLO`:

```text
What the model reads:   H E L L
What it should guess:  E L L O
```

The model tries to guess the next piece of text. When it gives the correct answer a low probability, its error increases. Training adjusts the model's numbers, called **weights**, to reduce that error.

It repeats this process on small passages from the books. During generation, it guesses one byte, adds it to the text, and guesses again.

A **byte** is a small unit of stored text. Basic English letters usually take one byte. Some characters, such as `é` and emoji, take several bytes. That is why a generated token is not always one visible character in this project.

## What is the difference between the methods?

| Method | Simple explanation |
|---|---|
| Transformer | Uses attention to gather information from earlier text |
| GRU | Updates a memory state as it reads the text |
| 4-gram | Counts short patterns and uses those counts to predict what comes next |

The Transformer and GRU learn weights. The 4-gram learns frequency counts. PyTorch supplies the tensor operations, automatic gradients, and GRU module; the project builds the model comparison and app around them.

## Questions you are likely to get

### Why use different books for training and testing?

> I wanted to check whether the models could predict a book they had not trained on. Three books were used for learning. A fourth helped choose which saved model to keep. A fifth was used for the final test.

The fourth book is the **validation set**. The fifth is the **test set**. Keeping their jobs separate helps avoid choosing a model just because it happens to score well on the final test.

### How did you make the comparison fair?

> The two neural models were almost the same size. They received the same sampled passages and the same total training-byte budget. I trained each model three times with different random seeds.

A **seed** sets the starting point for random choices. Different seeds help reveal whether a result depends on one lucky run. This comparison did not match training time or total computation, and the shared training settings may suit one model better than the other.

### What were the results?

> Lower scores were better. The Transformer averaged 2.730 bits per byte, the GRU averaged 2.505, and the 4-gram scored 2.295. So the GRU won between the neural models, while the simple counting method won overall.

**Bits per byte** measures how well the method predicts the actual text. It is not an accuracy percentage or a direct measure of how good the generated writing sounds.

### Why did the simple method win?

> Short patterns are useful for predicting this text, and counting learns them cheaply. The neural models had a small training budget. The result shows that these models and settings did not beat the baseline on this test.

This is a possible explanation, not proof of a universal rule. The 4-gram also counts all training text directly, whereas neural training samples passages repeatedly. More training and more test books would be useful follow-up experiments.

### What is attention?

> Attention gives different weights to earlier parts of the input when forming a prediction. In the app, I can select one attention head and see those weights.

The model must not see future text during training. A **causal mask** blocks those future positions. A test checks this by changing later text and verifying that earlier predictions stay the same. An attention picture shows the mixing weights; it does not fully explain why the model made a decision.

### What engineering problems did you handle?

- **Pausing training:** save the weights, optimizer state, and random-number states so a compatible run can continue correctly.
- **Choosing a model:** keep the version with the best validation score, even if the final training step is worse.
- **Checking files:** verify fingerprints of the books and saved models to detect changed files.
- **Reusing work:** cache earlier model computations while the input window grows. Rebuild the cache when the window slides to preserve the original predictions.

Choose one of these to explain in detail. The [developer guide](DEVELOPMENT.md) links to the relevant files.

### How did you test it?

> The tests check that the model cannot look ahead, that training passages stay inside one book, that evaluation counts the right targets, and that resumed training matches uninterrupted training on CPU. They also test saved model files, generation, and the API. The app was checked in Chromium and from a fresh checkout.

The project currently has **40 passing tests**. The [validation notes](VALIDATION.md) list what those checks cover.

### What would you improve next?

> I would test more books and authors, compare larger training budgets, and investigate why giving the Transformer more context did not consistently help. I would keep new final test data separate from model selection.

The current models produce rough text and have a short context window. Their scores do not show general reasoning or conversational ability.

## Give a short demo

1. Run `uv run --frozen python -m litforge serve` and open the app.
2. Show the result table, including the winning 4-gram baseline.
3. Generate from the same opening with both models.
4. Change the temperature and describe how the output changes.
5. Open Inside the model and explain the next-byte probabilities and attention picture.

If asked to show code, start with `litforge/model.py` or the training-recovery test in `tests/test_study.py`.

## Simple resume wording

> Built a PyTorch text-generation project comparing a Transformer, a GRU, and n-gram baselines using separate training, validation, and test books; repeated neural training across three seeds.

> Created a FastAPI app to compare model outputs and inspect predictions, with saved models, training recovery, and automated tests.

Use claims you can explain and demonstrate. The project uses established model designs; its work is in the implementation, comparison, testing, and app.
