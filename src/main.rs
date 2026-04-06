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

// ─── Demo ────────────────────────────────────────────────────────────────────

fn main() {
    // Synthetic 8×8 grayscale images.
    // "circle-like" pattern (bright centre, dark edges)
    #[rustfmt::skip]
    let circle_raw: Vec<u8> = vec![
         10,  20,  50,  80,  80,  50,  20,  10,
         20,  60, 120, 180, 180, 120,  60,  20,
         50, 120, 200, 240, 240, 200, 120,  50,
         80, 180, 240, 255, 255, 240, 180,  80,
         80, 180, 240, 255, 255, 240, 180,  80,
         50, 120, 200, 240, 240, 200, 120,  50,
         20,  60, 120, 180, 180, 120,  60,  20,
         10,  20,  50,  80,  80,  50,  20,  10,
    ];

    // "cross-like" pattern
    #[rustfmt::skip]
    let cross_raw: Vec<u8> = vec![
         10,  10,  10, 200, 200,  10,  10,  10,
         10,  10,  10, 200, 200,  10,  10,  10,
         10,  10,  10, 200, 200,  10,  10,  10,
        200, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200,
         10,  10,  10, 200, 200,  10,  10,  10,
         10,  10,  10, 200, 200,  10,  10,  10,
         10,  10,  10, 200, 200,  10,  10,  10,
    ];

    // "square-like" pattern (bright border, dark interior)
    #[rustfmt::skip]
    let square_raw: Vec<u8> = vec![
        220, 220, 220, 220, 220, 220, 220, 220,
        220,  20,  20,  20,  20,  20,  20, 220,
        220,  20,  20,  20,  20,  20,  20, 220,
        220,  20,  20,  20,  20,  20,  20, 220,
        220,  20,  20,  20,  20,  20,  20, 220,
        220,  20,  20,  20,  20,  20,  20, 220,
        220,  20,  20,  20,  20,  20,  20, 220,
        220, 220, 220, 220, 220, 220, 220, 220,
    ];

    let circle_tmpl = Image::from_raw(8, 8, circle_raw.clone());
    let cross_tmpl  = Image::from_raw(8, 8, cross_raw.clone());
    let square_tmpl = Image::from_raw(8, 8, square_raw.clone());

    // Build classifier with NCC metric, canonical size 8×8.
    let mut clf = TemplateClassifier::new(Metric::NCC, (8, 8));
    clf.add_template("circle", circle_tmpl);
    clf.add_template("cross",  cross_tmpl);
    clf.add_template("square", square_tmpl);

    println!("=== Template Matching Image Classifier ===\n");
    println!("Metric         : {}", clf.metric);
    println!("Canonical size : {}×{}\n", clf.canonical_size.0, clf.canonical_size.1);

    // ── Test 1: noisy circle (should classify as "circle") ──
    let noisy_circle: Vec<u8> = circle_raw.iter()
        .enumerate()
        .map(|(i, &p)| p.saturating_add(if i % 3 == 0 { 15 } else { 0 })
                        .saturating_sub(if i % 5 == 0 { 10 } else { 0 }))
        .collect();
    let query1 = Image::from_raw(8, 8, noisy_circle);

    println!("--- Query 1: noisy circle ---");
    let result1 = clf.classify(&query1);
    println!("{}", result1);

    // ── Test 2: noisy cross (should classify as "cross") ──
    let noisy_cross: Vec<u8> = cross_raw.iter()
        .enumerate()
        .map(|(i, &p)| p.saturating_add(if i % 7 == 0 { 20 } else { 0 }))
        .collect();
    let query2 = Image::from_raw(8, 8, noisy_cross);

    println!("--- Query 2: noisy cross ---");
    let result2 = clf.classify(&query2);
    println!("{}", result2);

    // ── Test 3: upscaled square (16×16 → resized to 8×8 for comparison) ──
    let big_square: Vec<u8> = {
        let small = Image::from_raw(8, 8, square_raw);
        let big   = small.resize(16, 16);
        big.data.iter().map(|&p| (p * 255.0) as u8).collect()
    };
    let query3 = Image::from_raw(16, 16, big_square);

    println!("--- Query 3: upscaled square (16×16) ---");
    let result3 = clf.classify(&query3);
    println!("{}", result3);

    // ── Test 4: same queries using SSD metric ──
    let mut clf_ssd = TemplateClassifier::new(Metric::SSD, (8, 8));
    clf_ssd.add_template("circle", Image::from_raw(8, 8, circle_raw.clone()));
    clf_ssd.add_template("cross",  Image::from_raw(8, 8, cross_raw.clone()));

    println!("--- Query 4: noisy circle using SSD metric ---");
    let result4 = clf_ssd.classify(&query1);
    println!("{}", result4);
}