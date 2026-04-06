use std::collections::HashMap;
use std::fmt;

// ─── Image Representation ───────────────────────────────────────────────────

/// A grayscale image stored as a flat Vec<f32> in row-major order.
/// Pixel values are normalized to [0.0, 1.0].
#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Image {
    /// Create a new image from raw pixel data (values 0–255).
    pub fn from_raw(width: usize, height: usize, raw: Vec<u8>) -> Self {
        assert_eq!(raw.len(), width * height, "Data size mismatch");
        let data = raw.iter().map(|&p| p as f32 / 255.0).collect();
        Image { width, height, data }
    }

    /// Create a blank (all-zero) image.
    pub fn zeros(width: usize, height: usize) -> Self {
        Image { width, height, data: vec![0.0; width * height] }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.width + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * self.width + col] = val;
    }

    /// Resize using nearest-neighbour interpolation.
    pub fn resize(&self, new_w: usize, new_h: usize) -> Image {
        let mut out = Image::zeros(new_w, new_h);
        for r in 0..new_h {
            for c in 0..new_w {
                let src_r = (r * self.height / new_h).min(self.height - 1);
                let src_c = (c * self.width / new_w).min(self.width - 1);
                out.set(r, c, self.get(src_r, src_c));
            }
        }
        out
    }

    /// Mean pixel value.
    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    /// Standard deviation.
    pub fn std_dev(&self) -> f32 {
        let m = self.mean();
        let variance = self.data.iter().map(|&p| (p - m).powi(2)).sum::<f32>()
            / self.data.len() as f32;
        variance.sqrt()
    }
}

// ─── Similarity Metrics ──────────────────────────────────────────────────────

/// Available similarity metrics for comparing an image to a template.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    /// Sum of Squared Differences (lower = more similar).
    SSD,
    /// Normalised Cross-Correlation (higher = more similar, range [-1, 1]).
    NCC,
    /// Mean Absolute Difference (lower = more similar).
    MAD,
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Metric::SSD => write!(f, "SSD"),
            Metric::NCC => write!(f, "NCC"),
            Metric::MAD => write!(f, "MAD"),
        }
    }
}

/// Compute the similarity score between two same-sized images.
/// For SSD and MAD, *lower* means more similar.
/// For NCC, *higher* means more similar.
pub fn compute_score(img: &Image, template: &Image, metric: Metric) -> f32 {
    assert_eq!(img.data.len(), template.data.len(), "Images must be the same size");

    match metric {
        Metric::SSD => img.data.iter()
            .zip(&template.data)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>(),

        Metric::MAD => img.data.iter()
            .zip(&template.data)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / img.data.len() as f32,

        Metric::NCC => {
            let img_mean = img.mean();
            let tmpl_mean = template.mean();
            let img_std = img.std_dev().max(1e-8);
            let tmpl_std = template.std_dev().max(1e-8);

            let n = img.data.len() as f32;
            img.data.iter()
                .zip(&template.data)
                .map(|(a, b)| ((a - img_mean) / img_std) * ((b - tmpl_mean) / tmpl_std))
                .sum::<f32>() / n
        }
    }
}

// ─── Classifier ──────────────────────────────────────────────────────────────

/// A single template associated with a class label.
#[derive(Clone, Debug)]
pub struct Template {
    pub label: String,
    pub image: Image,
}

/// The classifier holds one or more templates per class and uses the
/// chosen metric to assign a label to an unseen query image.
pub struct TemplateClassifier {
    pub templates: Vec<Template>,
    pub metric: Metric,
    /// The canonical size every image is rescaled to before comparison.
    pub canonical_size: (usize, usize), // (width, height)
}

impl TemplateClassifier {
    pub fn new(metric: Metric, canonical_size: (usize, usize)) -> Self {
        TemplateClassifier { templates: Vec::new(), metric, canonical_size }
    }

    /// Add a labelled template.  It is automatically resized to canonical_size.
    pub fn add_template(&mut self, label: impl Into<String>, image: Image) {
        let (w, h) = self.canonical_size;
        let resized = image.resize(w, h);
        self.templates.push(Template { label: label.into(), image: resized });
    }

    /// Classify a query image.
    /// Returns `(best_label, score, all_scores_per_class)`.
    pub fn classify(&self, query: &Image) -> ClassificationResult {
        assert!(!self.templates.is_empty(), "No templates registered");

        let (w, h) = self.canonical_size;
        let query_resized = query.resize(w, h);

        // Aggregate per-class best score.
        // For SSD/MAD: track minimum; for NCC: track maximum.
        let mut class_scores: HashMap<String, f32> = HashMap::new();

        for tmpl in &self.templates {
            let score = compute_score(&query_resized, &tmpl.image, self.metric);

            let entry = class_scores.entry(tmpl.label.clone()).or_insert(match self.metric {
                Metric::NCC => f32::NEG_INFINITY,
                _ => f32::INFINITY,
            });

            *entry = match self.metric {
                Metric::NCC => entry.max(score),
                _ => entry.min(score),
            };
        }

        // Pick best class.
        let best_label = class_scores.iter()
            .min_by(|(_, a), (_, b)| {
                match self.metric {
                    // For NCC we want the *highest* score, so invert comparison.
                    Metric::NCC => b.partial_cmp(a).unwrap(),
                    _ => a.partial_cmp(b).unwrap(),
                }
            })
            .map(|(label, _)| label.clone())
            .unwrap();

        let best_score = class_scores[&best_label];

        ClassificationResult {
            label: best_label,
            score: best_score,
            all_scores: class_scores,
            metric: self.metric,
        }
    }
}

// ─── Result Type ─────────────────────────────────────────────────────────────

pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
    pub all_scores: HashMap<String, f32>,
    pub metric: Metric,
}

impl fmt::Display for ClassificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Predicted class : {}", self.label)?;
        writeln!(f, "Best score ({})  : {:.6}", self.metric, self.score)?;
        writeln!(f, "All class scores:")?;

        let mut scores: Vec<_> = self.all_scores.iter().collect();
        scores.sort_by(|a, b| a.0.cmp(b.0));
        for (label, score) in scores {
            let marker = if label == &self.label { " ◄" } else { "" };
            writeln!(f, "  {:>12} : {:.6}{}", label, score, marker)?;
        }
        Ok(())
    }
}
use std::fs::File;
use std::io::{Read, BufReader};

fn read_u32_be(reader: &mut impl Read) -> u32 {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).unwrap();
    u32::from_be_bytes(buf)
}

pub fn load_mnist_images(path: &str) -> Vec<Image> {
    let f = File::open(path).expect("Cannot open image file");
    let mut r = BufReader::new(f);

    let magic = read_u32_be(&mut r);
    assert_eq!(magic, 0x0803, "Bad image file magic");

    let count  = read_u32_be(&mut r) as usize;
    let rows   = read_u32_be(&mut r) as usize;
    let cols   = read_u32_be(&mut r) as usize;

    let pixel_count = rows * cols;
    let mut images = Vec::with_capacity(count);

    for _ in 0..count {
        let mut raw = vec![0u8; pixel_count];
        r.read_exact(&mut raw).unwrap();
        images.push(Image::from_raw(cols, rows, raw));
    }
    images
}

pub fn load_mnist_labels(path: &str) -> Vec<u8> {
    let f = File::open(path).expect("Cannot open label file");
    let mut r = BufReader::new(f);

    let magic = read_u32_be(&mut r);
    assert_eq!(magic, 0x0801, "Bad label file magic");

    let count = read_u32_be(&mut r) as usize;
    let mut labels = vec![0u8; count];
    r.read_exact(&mut labels).unwrap();
    labels
}
pub fn build_mean_templates(
    images: &[Image],
    labels: &[u8],
    num_classes: usize,
) -> Vec<(String, Image)> {
    let (w, h) = (images[0].width, images[0].height);
    let pixel_count = w * h;

    let mut accum  = vec![vec![0f64; pixel_count]; num_classes];
    let mut counts = vec![0usize; num_classes];

    for (img, &label) in images.iter().zip(labels.iter()) {
        let cls = label as usize;
        counts[cls] += 1;
        for (acc, &px) in accum[cls].iter_mut().zip(img.data.iter()) {
            *acc += px as f64;
        }
    }

    (0..num_classes)
        .map(|cls| {
            let n = counts[cls] as f64;
            let mean_pixels: Vec<f32> = accum[cls].iter()
                .map(|&s| (s / n) as f32)
                .collect();
            let tmpl_image = Image { width: w, height: h, data: mean_pixels };
            (cls.to_string(), tmpl_image)
        })
        .collect()
}

// ─── Demo ────────────────────────────────────────────────────────────────────

fn main() {
    // --- Download and unzip MNIST from http://yann.lecun.com/exdb/mnist/
    // Place the four files in a `data/` folder next to Cargo.toml
    let train_images = load_mnist_images("data/train-images.idx3-ubyte");
    let train_labels = load_mnist_labels("data/train-labels.idx1-ubyte");
    let test_images  = load_mnist_images("data/t10k-images.idx3-ubyte");
    let test_labels  = load_mnist_labels("data/t10k-labels.idx1-ubyte");

    println!("Loaded {} train, {} test samples", train_images.len(), test_images.len());

    // Build one mean template per digit class (0–9)
    let templates = build_mean_templates(&train_images, &train_labels, 10);

    let mut clf = TemplateClassifier::new(Metric::NCC, (28, 28));
    for (label, img) in templates {
        clf.add_template(label, img);
    }

    // Evaluate
    let mut correct = 0usize;
    let total = test_images.len();

    for (i, (img, &true_label)) in test_images.iter().zip(test_labels.iter()).enumerate() {
        let result = clf.classify(img);
        let predicted: u8 = result.label.parse().unwrap();
        if predicted == true_label {
            correct += 1;
        }
        // Print progress every 1000 samples
        if (i + 1) % 1000 == 0 {
            println!("  [{}/{}]  running accuracy: {:.2}%",
                i + 1, total,
                100.0 * correct as f32 / (i + 1) as f32);
        }
    }

    println!("\nFinal accuracy: {}/{} = {:.2}%", correct, total,
        100.0 * correct as f32 / total as f32);
}