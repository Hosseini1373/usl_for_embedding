import mlflow
mlflow.set_tracking_uri("http://localhost:8080")

# Implicitly create a new experiment
mlflow.set_experiment("XOR")

epochs = 10000

with mlflow.start_run() as run:
    # Log the hyperparameters
    # Hyperparameters
    hp = {
        "activation": activation.__class__.__name__,
        "lr": 0.02,
        "momentum": 0.9,
        "epochs": epochs,
        "loss_fn": loss_fn.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
    }
    mlflow.log_params(hp)
    # Train the model

    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = loss_fn(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")
            mlflow.log_metric("loss", f"{loss:2f}", step=epoch)

    # Save the trained model to MLflow.
    mlflow.pytorch.log_model(model, "model")
