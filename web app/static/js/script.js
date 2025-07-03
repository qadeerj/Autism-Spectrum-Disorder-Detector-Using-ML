document.addEventListener("DOMContentLoaded", () => {
  // Neural Network Animation
  const neuralNetwork = document.getElementById("neural-network")
  if (neuralNetwork) {
    initNeuralNetwork(neuralNetwork)
  }

  // Mobile Navigation
  const mobileMenuBtn = document.querySelector(".mobile-menu-btn")
  const navMenu = document.querySelector(".nav-menu")

  if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener("click", () => {
      navMenu.classList.toggle("active")
    })
  }

  // Close mobile menu when clicking outside
  document.addEventListener("click", (e) => {
    if (
      navMenu &&
      navMenu.classList.contains("active") &&
      !e.target.closest(".nav-menu") &&
      !e.target.closest(".mobile-menu-btn")
    ) {
      navMenu.classList.remove("active")
    }
  })

  // Smooth scrolling for navigation links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault()

      if (navMenu && navMenu.classList.contains("active")) {
        navMenu.classList.remove("active")
      }

      const targetId = this.getAttribute("href")
      const targetElement = document.querySelector(targetId)

      if (targetElement) {
        const headerHeight = document.querySelector(".header").offsetHeight

        window.scrollTo({
          top: targetElement.offsetTop - headerHeight,
          behavior: "smooth",
        })
      }

      // Update active link
      document.querySelectorAll(".nav-menu a").forEach((link) => {
        link.classList.remove("active")
      })
      this.classList.add("active")
    })
  })

  // Update active link on scroll
  window.addEventListener("scroll", () => {
    const sections = document.querySelectorAll("section")
    const navLinks = document.querySelectorAll(".nav-menu a")
    const headerHeight = document.querySelector(".header").offsetHeight

    let current = ""

    sections.forEach((section) => {
      const sectionTop = section.offsetTop - headerHeight - 100
      const sectionHeight = section.clientHeight

      if (pageYOffset >= sectionTop && pageYOffset < sectionTop + sectionHeight) {
        current = "#" + section.getAttribute("id")
      }
    })

    navLinks.forEach((link) => {
      link.classList.remove("active")
      if (link.getAttribute("href") === current) {
        link.classList.add("active")
      }
    })
  })

  // Form handling
  const dataTypeSelect = document.getElementById("dataType")
  const analysisTypeGroup = document.getElementById("analysisTypeGroup")
  const analysisTypeSelect = document.getElementById("analysisType")
  const fileUploadGroup = document.getElementById("fileUploadGroup")
  const fileUpload = document.getElementById("fileUpload")
  const fileName = document.getElementById("file-name")
  const processButton = document.getElementById("processButton")
  const progressContainer = document.getElementById("progressContainer")
  const progress = document.getElementById("progress")
  const progressText = document.getElementById("progressText")
  const resultContainer = document.getElementById("resultContainer")
  const resultImage = document.getElementById("resultImage")
  const resultText = document.getElementById("resultText")
  const downloadButton = document.getElementById("downloadButton")
  const fusionUploadGroup = document.getElementById("fusionUploadGroup")

  // Handle data type selection
  if (dataTypeSelect) {
    dataTypeSelect.addEventListener("change", function () {
      const selectedValue = this.value

      // Reset form elements
      analysisTypeGroup.style.display = "none"
      fileUploadGroup.style.display = "none"
      fusionUploadGroup.style.display = "none"
      processButton.disabled = true

      if (selectedValue === "sMRI") {
        analysisTypeGroup.style.display = "block"
        analysisTypeSelect.innerHTML = `
                <option value="">-- Select Analysis --</option>
                <option value="Classification">Classification</option>
                <option value="ClassificationLocalization">Classification + Localization</option>
            `
      } else if (selectedValue === "fMRI") {
        analysisTypeGroup.style.display = "block"
        analysisTypeSelect.innerHTML = `
                <option value="">-- Select Analysis --</option>
                <option value="Classification">Classification</option>
            `
      } else if (selectedValue === "Fusion") {
        analysisTypeGroup.style.display = "block"
        analysisTypeSelect.innerHTML = `
                <option value="">-- Select Analysis --</option>
                <option value="Classification">Classification</option>
            `
      }
    })
  }

  // Handle analysis type selection
  if (analysisTypeSelect) {
    analysisTypeSelect.addEventListener("change", function () {
      const dataType = dataTypeSelect.value

      if (this.value) {
        if (dataType === "Fusion") {
          fusionUploadGroup.style.display = "block"
          fileUploadGroup.style.display = "none"
          processButton.disabled = true
        } else {
          fileUploadGroup.style.display = "block"
          fusionUploadGroup.style.display = "none"
          processButton.disabled = true
        }
      } else {
        fileUploadGroup.style.display = "none"
        fusionUploadGroup.style.display = "none"
        processButton.disabled = true
      }
    })
  }

  // Handle file upload
  if (fileUpload) {
    fileUpload.addEventListener("change", function () {
      if (this.files.length > 0) {
        fileName.textContent = this.files[0].name
        processButton.disabled = false
      } else {
        fileName.textContent = "No file chosen"
        processButton.disabled = true
      }
    })
  }

  // Handle sMRI file upload for Fusion
  const sMRIFileUpload = document.getElementById("sMRIFileUpload")
  const sMRIFileName = document.getElementById("sMRI-file-name")
  const fMRIFileUpload = document.getElementById("fMRIFileUpload")
  const fMRIFileName = document.getElementById("fMRI-file-name")
  let sMRIFileSelected = false
  let fMRIFileSelected = false

  if (sMRIFileUpload) {
    sMRIFileUpload.addEventListener("change", function () {
      if (this.files.length > 0) {
        sMRIFileName.textContent = this.files[0].name
        sMRIFileSelected = true

        // Enable process button if both files are selected
        if (fMRIFileSelected) {
          processButton.disabled = false
        }
      } else {
        sMRIFileName.textContent = "No file chosen"
        sMRIFileSelected = false
        processButton.disabled = true
      }
    })
  }

  // Handle fMRI file upload for Fusion
  if (fMRIFileUpload) {
    fMRIFileUpload.addEventListener("change", function () {
      if (this.files.length > 0) {
        fMRIFileName.textContent = this.files[0].name
        fMRIFileSelected = true

        // Enable process button if both files are selected
        if (sMRIFileSelected) {
          processButton.disabled = false
        }
      } else {
        fMRIFileName.textContent = "No file chosen"
        fMRIFileSelected = false
        processButton.disabled = true
      }
    })
  }

  // Handle process button click
// Handle the process button click
if (processButton) {
  processButton.addEventListener("click", () => {
    const dataType = dataTypeSelect.value;
    const analysisType = analysisTypeSelect.value;

    // Create FormData object to send file(s) to the server
    const formData = new FormData();
    formData.append("dataType", dataType);
    formData.append("analysisType", analysisType);

    // Check if files are selected based on dataType
    if (dataType === "Fusion") {
      const sMRIFile = sMRIFileUpload.files[0];
      const fMRIFile = fMRIFileUpload.files[0];
      if (!sMRIFile || !fMRIFile) {
        alert("Please complete all fields before processing.");
        return;
      }
      formData.append("sMRIFile", sMRIFile);
      formData.append("fMRIFile", fMRIFile);
    } else {
      const file = fileUpload.files[0];
      if (!file) {
        alert("Please complete all fields before processing.");
        return;
      }
      formData.append("file", file);
    }

    // Show progress bar
    progressContainer.style.display = "block";
    resultContainer.style.display = "none";  // Hide result container initially
    progress.style.width = "0%";  // Reset the progress bar to 0%

    // Start a setInterval to simulate progress bar animation
    let progressValue = 0;
    const progressInterval = setInterval(() => {
      progressValue += 5; // Increment progress by 5%
      progress.style.width = progressValue + "%"; // Update progress bar width
      progressText.textContent = `Processing... ${progressValue}%`;

      if (progressValue >= 100) {
        clearInterval(progressInterval); // Stop the interval once progress reaches 100%
        progressText.textContent = "Processing complete!";
      }
    }, 200); // Update progress every 200ms

    // Make the actual request to the backend
    fetch('/process', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())  // Parse the JSON response
    .then(data => {
      // Once backend responds, complete the progress bar at 100%
      progress.style.width = "100%"; // Ensure progress bar completes at 100%

      // Call displayResults with the actual data from the backend
      displayResults(data); // Update UI with actual server results
    })
    .catch(error => {
      console.error('Error:', error);
      alert('An error occurred during processing.');
    });
  });
}

// Function to display results dynamically
function displayResults(data) {
  if (data && data.prediction && data.confidence && data.result_image && data.model_used) {
    // Update the result section with the actual data
    resultText.innerHTML = `
      <p><strong>Prediction:</strong> ${data.prediction}</p>
      <p><strong>Confidence:</strong> ${data.confidence}%</p>
      <p><strong>Model Used:</strong> ${data.model_used}</p>
    `;
    
    // Set the result image source dynamically from the backend response
    resultImage.src = data.result_image;
    
    // Show the result container (this will make the results visible)
    resultContainer.style.display = "block";

    // Optionally, scroll to the results section
    setTimeout(() => {
      resultContainer.scrollIntoView({ behavior: "smooth" });
    }, 300);
  } else {
    alert("Received invalid data from the server.");
  }
}

  
  // Handle download button
  if (downloadButton) {
    downloadButton.addEventListener("click", () => {
      // In a real application, this would trigger a download of the results
      alert("In a production environment, this would download the full analysis results as a PDF or image file.")
    })
  }

  // Contact form submission
  const contactForm = document.getElementById("contactForm")

  if (contactForm) {
    contactForm.addEventListener("submit", (e) => {
      e.preventDefault()

      const name = document.getElementById("name").value
      const email = document.getElementById("email").value
      const message = document.getElementById("message").value

      // In a real application, you would send this data to the server
      // For demonstration purposes, we'll just show an alert
      alert(`Thank you, ${name}! Your message has been received. We'll get back to you at ${email} soon.`)

      // Reset the form
      contactForm.reset()
    })
  }
})

// Neural Network Animation - Modified to be more brain-like
function initNeuralNetwork(container) {
  // Canvas setup
  const canvas = document.createElement("canvas")
  const ctx = canvas.getContext("2d")
  container.appendChild(canvas)

  // Set canvas size
  function resizeCanvas() {
    canvas.width = container.offsetWidth
    canvas.height = container.offsetHeight
  }

  resizeCanvas()
  window.addEventListener("resize", resizeCanvas)

  // Brain shape parameters
  const centerX = canvas.width / 2
  const centerY = canvas.height / 2
  const brainWidth = Math.min(canvas.width * 0.8, 800)
  const brainHeight = Math.min(canvas.height * 0.7, 600)

  // Neuron class
  class Neuron {
    constructor(x, y) {
      this.x = x
      this.y = y
      this.originalX = x
      this.originalY = y
      this.connections = []
      this.size = Math.random() * 3 + 2
      this.speed = Math.random() * 0.3 + 0.1
      this.angle = Math.random() * Math.PI * 2
      this.pulseSize = 0
      this.pulseSpeed = Math.random() * 0.05 + 0.02
      this.pulsing = false
      this.pulseDelay = Math.random() * 200
      this.pulseTimeout = null
      this.maxDistance = Math.random() * 20 + 10 // Max distance from original position
    }

    update() {
      // Move in a random direction but stay close to original position
      this.x += Math.cos(this.angle) * this.speed
      this.y += Math.sin(this.angle) * this.speed

      // Calculate distance from original position
      const dx = this.x - this.originalX
      const dy = this.y - this.originalY
      const distance = Math.sqrt(dx * dx + dy * dy)

      // If too far, move back toward original position
      if (distance > this.maxDistance) {
        this.angle = Math.atan2(this.originalY - this.y, this.originalX - this.x)
        this.speed = Math.random() * 0.5 + 0.3 // Faster return speed
      } else if (Math.random() < 0.02) {
        // Occasionally change direction
        this.angle = Math.random() * Math.PI * 2
        this.speed = Math.random() * 0.3 + 0.1
      }

      // Update pulse
      if (this.pulsing) {
        this.pulseSize += this.pulseSpeed
        if (this.pulseSize > 20) {
          this.pulsing = false
          this.pulseSize = 0

          // Schedule next pulse
          this.pulseTimeout = setTimeout(
            () => {
              this.pulsing = true
            },
            Math.random() * 5000 + 2000,
          )
        }
      }
    }

    draw() {
      // Draw neuron
      ctx.beginPath()
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2)
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--neuron-color")
      ctx.fill()

      // Draw pulse
      if (this.pulsing && this.pulseSize > 0) {
        ctx.beginPath()
        ctx.arc(this.x, this.y, this.pulseSize, 0, Math.PI * 2)
        ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--neuron-color")
        ctx.lineWidth = 1
        ctx.stroke()
      }
    }

    connect(neuron) {
      this.connections.push(neuron)
    }

    drawConnections() {
      this.connections.forEach((neuron) => {
        const distance = Math.sqrt(Math.pow(this.x - neuron.x, 2) + Math.pow(this.y - neuron.y, 2))
        if (distance < 150) {
          ctx.beginPath()
          ctx.moveTo(this.x, this.y)
          ctx.lineTo(neuron.x, neuron.y)
          ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--synapse-color")
          ctx.lineWidth = 1 - distance / 150
          ctx.stroke()
        }
      })
    }

    startPulsing() {
      setTimeout(() => {
        this.pulsing = true
      }, this.pulseDelay)
    }
  }

  // Create neurons in a brain-like shape
  const neurons = []
  const neuronCount = Math.min(120, Math.floor((canvas.width * canvas.height) / 8000))

  // Helper function to check if a point is within the brain shape
  function isInBrainShape(x, y) {
    // Normalize coordinates to -1 to 1 range
    const nx = (x - centerX) / (brainWidth / 2)
    const ny = (y - centerY) / (brainHeight / 2)

    // Brain shape equation (approximation of brain outline)
    const leftHemisphere = Math.pow(nx + 0.5, 2) + Math.pow(ny, 2) < 0.8
    const rightHemisphere = Math.pow(nx - 0.5, 2) + Math.pow(ny, 2) < 0.8
    const middleDivide = Math.abs(nx) < 0.1 && Math.abs(ny) < 0.8

    return (leftHemisphere || rightHemisphere) && !middleDivide
  }

  // Create neurons in brain shape
  for (let i = 0; i < neuronCount * 2; i++) {
    const x = centerX + (Math.random() * 2 - 1) * (brainWidth / 2)
    const y = centerY + (Math.random() * 2 - 1) * (brainHeight / 2)

    if (isInBrainShape(x, y) && neurons.length < neuronCount) {
      neurons.push(new Neuron(x, y))
    }
  }

  // Connect neurons
  neurons.forEach((neuron) => {
    const connectionCount = Math.floor(Math.random() * 3) + 2
    for (let i = 0; i < connectionCount; i++) {
      const randomNeuron = neurons[Math.floor(Math.random() * neurons.length)]
      if (randomNeuron !== neuron) {
        neuron.connect(randomNeuron)
      }
    }
    neuron.startPulsing()
  })

  // Animation loop
  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Update and draw connections
    neurons.forEach((neuron) => {
      neuron.update()
      neuron.drawConnections()
    })

    // Draw neurons on top
    neurons.forEach((neuron) => {
      neuron.draw()
    })

    requestAnimationFrame(animate)
  }

  animate()
}
