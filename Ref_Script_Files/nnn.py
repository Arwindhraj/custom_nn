import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiscaleObjectDetectionModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(MultiscaleObjectDetectionModel, self).__init__()

        # Branch 1: Convolutional Block with Kernel Size 3x3
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.silu1 = nn.SiLU()

        # Branch 2: Convolutional Block with Kernel Size 5x5
        self.conv2 = nn.Conv2d(input_channels, 256, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.silu2 = nn.SiLU()

        # Branch 3: Convolutional Block with Kernel Size 7x7
        self.conv3 = nn.Conv2d(input_channels, 256, kernel_size=7, stride=1, padding=3)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.silu3 = nn.SiLU()

        # Additional Convolutional Blocks
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.silu4 = nn.SiLU()

        # Transformer Layer
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)

        # Output Layer
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        # Branch 1
        out1 = self.silu1(self.batchnorm1(self.conv1(x))) + x

        # Branch 2
        out2 = self.silu2(self.batchnorm2(self.conv2(x))) + x

        # Branch 3
        out3 = self.silu3(self.batchnorm3(self.conv3(x))) + x

        # Merge Multiscale Features
        multiscale_features = torch.cat([out1, out2, out3], dim=1)

        # Additional Convolutional Blocks
        out4 = self.silu4(self.batchnorm4(self.conv4(multiscale_features))) + x

        # Transformer Layer
        transformer_output = self.transformer(out4.view(out4.size(0), out4.size(1), -1))

        # Output Layer
        final_output = self.output_layer(transformer_output.mean(dim=2))

        return final_output

# Instantiate the model
input_channels = 3  # Adjust based on the number of input channels
num_classes = 20  # Adjust based on the number of object classes
model = MultiscaleObjectDetectionModel(input_channels, num_classes)

# Print the model architecture
print(model)

#################################################


# Instantiate the model
input_channels = 3
num_classes = 20
model = MultiscaleObjectDetectionModel(input_channels, num_classes)

# Loss functions
classification_criterion = nn.CrossEntropyLoss()
localization_criterion = nn.SmoothL1Loss()
objectness_criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop (pseudo-code, you may need to adapt based on your data loading)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for inputs, targets_classification, targets_localization, targets_objectness in train_dataloader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass

        # Assuming your model returns classification, localization, and objectness outputs
        classification_output = outputs['classification']
        localization_output = outputs['localization']
        objectness_output = outputs['objectness']

        # Compute losses
        loss_classification = classification_criterion(classification_output, targets_classification)
        loss_localization = localization_criterion(localization_output, targets_localization)
        loss_objectness = objectness_criterion(objectness_output, targets_objectness)

        # Combine losses with appropriate weights
        weight_classification = 1.0
        weight_localization = 1.0
        weight_objectness = 1.0
        total_loss = (weight_classification * loss_classification +
                      weight_localization * loss_localization +
                      weight_objectness * loss_objectness)

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

    # Print or log training loss for monitoring
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}')

# Save the trained model if needed
torch.save(model.state_dict(), 'multiscale_object_detection_model.pth')
