// Global functions
// Conv layers
function addConvLayer() {
  const container = document.getElementById("conv-layers-container");
  if (!container) return;

  const newLayer = document.createElement("div");
  newLayer.className = "conv-layer mb-3";
  newLayer.innerHTML = `
      <div class="row">
        <div class="col-md-3">
          <label>Filters:</label>
          <input type="number" name="filters[]" min="16" max="512" value="32" class="form-control" required>
        </div>
        <div class="col-md-3">
          <label>Kernel size:</label>
          <select name="kernel_size[]" class="form-control">
            <option value="3">3x3 (standard)</option>
            <option value="5">5x5</option>
            <option value="7">7x7</option>
          </select>
        </div>
        <div class="col-md-3">
          <label>Padding:</label>
          <select name="padding[]" class="form-control">
            <option value="same">Same (preserve size)</option>
            <option value="valid">Valid (reduce size)</option>
          </select>
        </div>
        <div class="col-md-3">
          <label>Activation:</label>
          <select name="activation[]" class="form-control">
            <option value="relu">ReLU</option>
            <option value="leaky_relu">Leaky ReLU</option>
            <option value="elu">ELU</option>
          </select>
        </div>
      </div>
    `;
  container.appendChild(newLayer);
}

// Dense layers
function addDenseLayer() {
  let container = document.getElementById("dense-layers-container");
  if (!container) {
    container = document.getElementById("dense-layer-container");
  }
  if (!container) return;

  const newLayer = document.createElement("div");
  newLayer.className = "dense-layer mb-3";
  newLayer.innerHTML = `
      <label>Units:</label>
      <input type="number" name="dense_units[]" min="32" max="1024" value="128" class="form-control" required>
    `;
  container.appendChild(newLayer);
}

// Update classes
function updateClassList(classes) {
  const classList = document.getElementById("uploaded-class-list");
  if (!classList) return;

  if (classes.length > 0) {
    classList.innerHTML = `
          <div class="row">
            ${classes
              .map(
                (c) => `
              <div class="col-md-6 mb-3">
                <div class="class-card">
                  <div class="class-header">
                    <i class="fas fa-folder-open"></i>
                    <h5>${c.name}</h5>
                  </div>
                  <div class="class-stats">
                    <span class="badge badge-info">${c.count} images</span>
                  </div>
                  <button class="btn btn-sm btn-danger remove-class" 
                         data-class-name="${c.name}">
                    <i class="fas fa-trash"></i> Remove Class
                  </button>
                </div>
              </div>
            `
              )
              .join("")}
          </div>
          
          ${
            classes.length >= 2
              ? `<button id="continue-btn" class="btn btn-success mt-3" onclick="completeStep1()">
                Continue to Preprocessing
              </button>`
              : `<p class="text-info mt-3">Please upload at least 2 classes to continue.</p>`
          }
        `;
  } else {
    classList.innerHTML =
      '<p class="text-muted" id="no-classes-message">No classes uploaded yet. Upload at least 2 classes.</p>';
  }
}

// Step navigation current step
function completeStep1() {
  fetch("/update-step", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ step: 2 }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.status === "success") {
        window.location.href = "/preprocess";
      } else {
        alert(data.message || "Error updating step");
      }
    });
}

// Collapsible / Toggle
function toggleSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (!section) return;

  const content = section.querySelector(".collapsible-content");
  const checkmark = section.querySelector(".checkmark");
  const definitions = section.querySelectorAll(".definition");

  if (content) {
    const isOpening = !content.classList.contains("visible");
    content.classList.toggle("visible");

    if (!isOpening) {
      definitions.forEach((def) => {
        def.style.display = "none";
      });
    }

    if (!section.dataset.opened) {
      section.dataset.opened = "true";
      if (checkmark) {
        checkmark.classList.toggle("checked");
      }
    }
  }
}

function handleInfoToggleClick(e) {
  if (e.target.closest(".info-toggle")) {
    e.preventDefault();
    e.stopPropagation();

    const toggle = e.target.closest(".info-toggle");
    const targetId = toggle.dataset.target;
    const targetElement = document.getElementById(targetId);

    if (targetElement) {
      if (targetElement.style.display === "none") {
        targetElement.style.display = "block";
      } else {
        targetElement.style.display = "none";
      }

      const icon = toggle.querySelector("i");
      if (icon) {
        icon.classList.toggle("fa-question-circle");
        icon.classList.toggle("fa-chevron-up");
      }
    }
  }
}

function toggleDefinition(e, definitionId) {
  const definition = document.getElementById(definitionId);
  if (!definition) return;

  e.preventDefault();
  if (!definition.closest(".collapsible-section")) {
    definition.style.display =
      definition.style.display === "none" ? "block" : "none";
    return;
  }

  const section = definition.closest(".collapsible-section");
  const content = section
    ? section.querySelector(".collapsible-content")
    : null;

  if (content && content.classList.contains("visible")) {
    if (definition.style.display === "none") {
      definition.style.display = "block";
    } else {
      definition.style.display = "none";
    }
  } else if (section) {
    definition.style.display = "none";
  }
}

// Remove class
function removeClass(className) {
  if (confirm(`Are you sure you want to remove class "${className}"?`)) {
    fetch(`/remove-class/${className}`, {
      method: "POST",
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.status === "success") {
          updateClassList(data.classes);
        } else {
          alert(data.message || "Error removing class");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("Error removing class");
      });
  }
}

// Clear classes
function clearAllClasses() {
  if (confirm("Are you sure you want to remove all the classes?")) {
    fetch("/clear-classes", {
      method: "POST",
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.status === "success") {
          updateClassList([]);
          document.getElementById("clear-classes-btn").style.display = "none";
          alert("All classes removed!.");
        } else {
          alert(data.message || "Error removing classes");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("Error removing classes");
      });
  }
}

// Main DOM
document.addEventListener("DOMContentLoaded", function () {
  const trainSplitInput = document.getElementById("train-split");
  const valSplitInput = document.getElementById("val-split");
  const testSplitInput = document.getElementById("test-split");
  const fileInputs = document.querySelectorAll(".file-input");
  const customDataBtn = document.getElementById("custom-data-btn");
  const preCreatedDataBtn = document.getElementById("preCreated-data-btn");
  const customUploadForm = document.getElementById("custom-upload-form");
  const preCreatedDatasetForm = document.getElementById(
    "preCreated-dataset-form"
  );
  const clickableTerms = document.querySelectorAll(".clickable-term");
  const datasetCards = document.querySelectorAll(".dataset-card-wrapper");

  document.addEventListener("click", handleInfoToggleClick);

  // Dataset toggle
  customDataBtn.addEventListener("click", function () {
    customDataBtn.classList.add("active");
    customDataBtn.classList.remove("btn-outline-primary");
    customDataBtn.classList.add("btn-primary");
    preCreatedDataBtn.classList.remove("active");
    preCreatedDataBtn.classList.add("btn-outline-primary");
    preCreatedDataBtn.classList.remove("btn-primary");

    customUploadForm.style.display = "block";
    preCreatedDatasetForm.style.display = "none";
    document.querySelector(".uploaded-classes").style.display = "block";
    document.getElementById("clear-classes-btn").style.display = "none";
  });

  preCreatedDataBtn.addEventListener("click", function () {
    preCreatedDataBtn.classList.add("active");
    preCreatedDataBtn.classList.remove("btn-outline-primary");
    preCreatedDataBtn.classList.add("btn-primary");
    customDataBtn.classList.remove("active");
    customDataBtn.classList.add("btn-outline-primary");
    customDataBtn.classList.remove("btn-primary");

    customUploadForm.style.display = "none";
    preCreatedDatasetForm.style.display = "block";
    document.querySelector(".uploaded-classes").style.display = "none";

    const uploadedClasses = document.querySelectorAll(".class-card");
    if (uploadedClasses.length > 0) {
      document.getElementById("clear-classes-btn").style.display = "block";
    }
  });

  // Files
  fileInputs.forEach((fileInput) => {
    fileInput.addEventListener("change", function (e) {
      const files = Array.from(e.target.files);
      const imageFiles = files.filter((file) => file.type.startsWith("image/"));

      if (files.length === 0) return;

      const folderName = files[0].webkitRelativePath.split("/")[0];
      const folderInfo = document.querySelector(".selected-folder-info");

      if (folderInfo) {
        folderInfo.style.display = "block";

        const folderNameEl = document.querySelector("#folder-name");
        if (folderNameEl) folderNameEl.textContent = `Folder: ${folderName}`;

        const imageCountEl = document.querySelector("#image-count");
        if (imageCountEl)
          imageCountEl.textContent = `Containing ${imageFiles.length} images`;

        const classNameEl = document.querySelector("#class-name");
        if (classNameEl) classNameEl.value = folderName;
      }
    });
  });

  // Upload form submission handler
  const uploadForm = document.querySelector(".upload-form");
  if (uploadForm) {
    uploadForm.addEventListener("submit", function (e) {
      e.preventDefault();

      const formData = new FormData(this);
      fetch("/upload", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.status === "success") {
            uploadForm.reset();
            const folderInfo = document.querySelector(".selected-folder-info");
            if (folderInfo) folderInfo.style.display = "none";
            document.querySelector(".uploaded-classes").style.display = "block";

            const classListContainer = document.getElementById(
              "uploaded-class-list"
            );
            if (classListContainer) {
              updateClassList(data.classes);
            }

            alert(data.message);
          } else {
            alert(data.message || "Error uploading class");
          }
        })
        .catch((error) => {
          console.error("Error:", error);
          alert("Error uploading class folder");
        });
    });
  }

  // Remove class button handler
  document.addEventListener("click", function (e) {
    const removeButton = e.target.closest(".remove-class");
    if (removeButton) {
      const className = removeButton.dataset.className;
      removeClass(className);
    }
  });

  // Show the custom size fields
  const imageSizeSelect = document.getElementById("image-size");
  const customSizeFields = document.getElementById("custom-size-fields");

  if (imageSizeSelect && customSizeFields) {
    imageSizeSelect.addEventListener("change", function () {
      if (this.value === "custom") {
        customSizeFields.style.display = "block";
      } else {
        customSizeFields.style.display = "none";
      }
    });
  }

  // Preprocessing split
  function updateSplits() {
    if (!trainSplitInput || !valSplitInput || !testSplitInput) return;

    const trainSplit = parseInt(trainSplitInput.value);
    const remainingSplit = 100 - trainSplit;

    if (remainingSplit >= 0) {
      const valSplit = Math.floor(remainingSplit / 2);
      const testSplit = remainingSplit - valSplit;

      valSplitInput.value = valSplit;
      testSplitInput.value = testSplit;
    }
  }

  if (trainSplitInput) {
    trainSplitInput.addEventListener("input", updateSplits);
    updateSplits();
  }

  // Clickable terms
  clickableTerms.forEach(function (term) {
    term.addEventListener("click", function () {
      var definitionId = this.getAttribute("data-definition");
      var definition = document.getElementById(definitionId);
      const allDefinitions = document.querySelectorAll(".definition.show");
      allDefinitions.forEach(function (openDef) {
        if (openDef.id !== definitionId) {
          openDef.classList.remove("show");
        }
      });
      definition.classList.toggle("show");
    });
  });

  // Dataset cards
  datasetCards.forEach((cardWrapper) => {
    cardWrapper.addEventListener("click", function () {
      const radio = this.querySelector('input[type="radio"]');
      radio.checked = true;

      document.querySelectorAll(".dataset-card").forEach((card) => {
        card.classList.remove("selected");
      });
      this.querySelector(".dataset-card").classList.add("selected");
    });
  });

  if (preCreatedDatasetForm) {
    preCreatedDatasetForm.addEventListener("submit", function (e) {
      const selectedDataset = document.querySelector(
        'input[name="dataset"]:checked'
      );

      if (!selectedDataset) {
        e.preventDefault();
        alert("Please select a dataset before continuing.");
        return false;
      }
    });
  }
});
