use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

// ─── Constants ───────────────────────────────────────────────────────────────

const TEMPLATES_PER_CLASS: usize = 10;
const KMEANS_ITERS: usize = 20;

// ─── Image ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn from_raw(width: usize, height: usize, raw: Vec<u8>) -> Self {
        assert_eq!(raw.len(), width * height, "Data size mismatch");
        let data = raw.iter().map(|&p| p as f32 / 255.0).collect();
        Image { width, height, data }
    }

    pub fn zeros(width: usize, height: usize) -> Self {
        Image { width, height, data: vec![0.0; width * height] }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.width + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.width + c] = v;
    }

    pub fn resize(&self, new_w: usize, new_h: usize) -> Image {
        let mut out = Image::zeros(new_w, new_h);
        for r in 0..new_h {
            for c in 0..new_w {
                let sr = (r * self.height / new_h).min(self.height - 1);
                let sc = (c * self.width / new_w).min(self.width - 1);
                out.set(r, c, self.get(sr, sc));
            }
        }
        out
    }

    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    pub fn std_dev(&self) -> f32 {
        let m = self.mean();
        let var = self.data.iter()
            .map(|&p| (p - m).powi(2))
            .sum::<f32>() / self.data.len() as f32;
        var.sqrt()
    }

    pub fn ssd_to(&self, other: &Image) -> f32 {
        self.data.iter().zip(&other.data)
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }
}

// ─── Metric ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric { SSD, NCC, MAD }

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Metric::SSD => write!(f, "SSD"),
            Metric::NCC => write!(f, "NCC"),
            Metric::MAD => write!(f, "MAD"),
        }
    }
}

pub fn compute_score(img: &Image, tmpl: &Image, metric: Metric) -> f32 {
    match metric {
        Metric::SSD => img.data.iter().zip(&tmpl.data)
            .map(|(a, b)| (a - b).powi(2)).sum(),

        Metric::MAD => img.data.iter().zip(&tmpl.data)
            .map(|(a, b)| (a - b).abs()).sum::<f32>() / img.data.len() as f32,

        Metric::NCC => {
            let im = img.mean();
            let tm = tmpl.mean();
            let is = img.std_dev().max(1e-8);
            let ts = tmpl.std_dev().max(1e-8);
            let n = img.data.len() as f32;
            img.data.iter().zip(&tmpl.data)
                .map(|(a, b)| ((a - im) / is) * ((b - tm) / ts))
                .sum::<f32>() / n
        }
    }
}

// ─── Classifier ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Template {
    pub label: u8,
    pub image: Image,
}

pub struct TemplateClassifier {
    pub templates: Vec<Template>,
    pub metric: Metric,
    pub canonical_size: (usize, usize),
}

impl TemplateClassifier {
    pub fn new(metric: Metric, size: (usize, usize)) -> Self {
        Self { templates: Vec::new(), metric, canonical_size: size }
    }

    pub fn add_template(&mut self, label: u8, image: Image) {
        let (w, h) = self.canonical_size;
        self.templates.push(Template { label, image: image.resize(w, h) });
    }

    pub fn classify(&self, query: &Image) -> u8 {
        let (w, h) = self.canonical_size;
        let q = query.resize(w, h);
        let mut best_label = 0;
        let mut best_score = match self.metric {
            Metric::NCC => f32::NEG_INFINITY,
            _ => f32::INFINITY,
        };
        for tmpl in &self.templates {
            let score = compute_score(&q, &tmpl.image, self.metric);
            let better = match self.metric {
                Metric::NCC => score > best_score,
                _ => score < best_score,
            };
            if better { best_score = score; best_label = tmpl.label; }
        }
        best_label
    }
}

// ─── MNIST Loading ───────────────────────────────────────────────────────────

fn read_u32_be(r: &mut impl Read) -> u32 {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).unwrap();
    u32::from_be_bytes(buf)
}

pub fn load_mnist_images(path: &str) -> Vec<Image> {
    let mut r = BufReader::new(File::open(path).expect("Cannot open image file"));
    assert_eq!(read_u32_be(&mut r), 0x0803, "Bad image magic");
    let count = read_u32_be(&mut r) as usize;
    let rows  = read_u32_be(&mut r) as usize;
    let cols  = read_u32_be(&mut r) as usize;
    (0..count).map(|_| {
        let mut raw = vec![0u8; rows * cols];
        r.read_exact(&mut raw).unwrap();
        Image::from_raw(cols, rows, raw)
    }).collect()
}

pub fn load_mnist_labels(path: &str) -> Vec<u8> {
    let mut r = BufReader::new(File::open(path).expect("Cannot open label file"));
    assert_eq!(read_u32_be(&mut r), 0x0801, "Bad label magic");
    let count = read_u32_be(&mut r) as usize;
    let mut labels = vec![0u8; count];
    r.read_exact(&mut labels).unwrap();
    labels
}

// ─── Template Builders ───────────────────────────────────────────────────────

pub fn build_mean_templates(images: &[Image], labels: &[u8], num_classes: usize) -> Vec<(u8, Image)> {
    let (w, h) = (images[0].width, images[0].height);
    let mut accum  = vec![vec![0f64; w * h]; num_classes];
    let mut counts = vec![0usize; num_classes];
    for (img, &label) in images.iter().zip(labels) {
        counts[label as usize] += 1;
        for (a, &p) in accum[label as usize].iter_mut().zip(&img.data) {
            *a += p as f64;
        }
    }
    (0..num_classes).map(|c| {
        let n = counts[c] as f64;
        let data = accum[c].iter().map(|&x| (x / n) as f32).collect();
        (c as u8, Image { width: w, height: h, data })
    }).collect()
}

pub fn kmeans(images: &[Image], k: usize, max_iters: usize) -> Vec<Image> {
    assert!(images.len() >= k);
    let (w, h) = (images[0].width, images[0].height);
    let pixels = w * h;
    let step = images.len() / k;
    let mut centroids: Vec<Image> = (0..k).map(|i| images[i * step].clone()).collect();
    let mut assignments = vec![0usize; images.len()];

    for _ in 0..max_iters {
        let new_assignments: Vec<usize> = images.par_iter()
            .map(|img| {
                centroids.iter().enumerate()
                    .map(|(ci, c)| (ci, img.ssd_to(c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap().0
            })
            .collect();

        let changed = new_assignments != assignments;
        assignments = new_assignments;
        if !changed { break; }

        let mut accum = vec![vec![0f64; pixels]; k];
        let mut counts = vec![0usize; k];
        for (img, &ci) in images.iter().zip(&assignments) {
            counts[ci] += 1;
            for (a, &p) in accum[ci].iter_mut().zip(&img.data) { *a += p as f64; }
        }
        for (ci, centroid) in centroids.iter_mut().enumerate() {
            if counts[ci] == 0 {
                *centroid = images[ci * step % images.len()].clone();
            } else {
                let n = counts[ci] as f64;
                centroid.data = accum[ci].iter().map(|&x| (x / n) as f32).collect();
            }
        }
    }
    centroids
}

pub fn build_kmeans_templates(
    images: &[Image], labels: &[u8],
    num_classes: usize, templates_per_class: usize, kmeans_iters: usize,
) -> Vec<(u8, Image)> {
    let mut by_class: HashMap<u8, Vec<&Image>> = HashMap::new();
    for (img, &label) in images.iter().zip(labels) {
        by_class.entry(label).or_default().push(img);
    }
    (0..num_classes as u8).into_par_iter()
        .flat_map(|label| {
            let class_images: Vec<Image> = by_class[&label].iter().map(|&i| i.clone()).collect();
            let k = templates_per_class.min(class_images.len());
            kmeans(&class_images, k, kmeans_iters)
                .into_par_iter()
                .map(move |c| (label, c))
        })
        .collect()
}

// ─── Evaluation Modes ────────────────────────────────────────────────────────

fn run_sequential(test_images: &[Image], test_labels: &[u8], clf: &TemplateClassifier) {
    println!("\n[Mode 1: Sequential — mean templates, single thread]");
    let total = test_images.len();
    let start = Instant::now();
    let mut correct = 0usize;
    for (i, (img, &label)) in test_images.iter().zip(test_labels).enumerate() {
        if clf.classify(img) == label { correct += 1; }
        if (i + 1) % 1000 == 0 {
            print!("\r  [{}/{}] running accuracy: {:.2}%   ",
                i + 1, total, 100.0 * correct as f32 / (i + 1) as f32);
            io::stdout().flush().unwrap();
        }
    }
    println!();
    print_results(correct, total, start.elapsed(), 1, "sequential / mean templates");
}

fn run_parallel_mean(test_images: &[Image], test_labels: &[u8], clf: Arc<TemplateClassifier>) {
    println!("\n[Mode 2: Parallel — mean templates, Rayon par_iter]");
    let pool = ThreadPoolBuilder::new().num_threads(num_cpus::get()).build().unwrap();
    let total = test_images.len();
    let start = Instant::now();
    let correct = pool.install(|| {
        test_images.par_iter().zip(test_labels.par_iter())
            .map(|(img, &label)| if clf.classify(img) == label { 1 } else { 0 })
            .sum::<usize>()
    });
    print_results(correct, total, start.elapsed(), pool.current_num_threads(),
        "parallel / mean templates");
}

fn run_parallel_kmeans(
    train_images: &[Image], train_labels: &[u8],
    test_images:  &[Image], test_labels:  &[u8],
) {
    println!("\n[Mode 3: Parallel — k-means templates, Rayon par_iter]");
    println!("  Building {} templates/class via k-means ({} iters)...",
        TEMPLATES_PER_CLASS, KMEANS_ITERS);
    let t0 = Instant::now();
    let templates = build_kmeans_templates(
        train_images, train_labels, 10, TEMPLATES_PER_CLASS, KMEANS_ITERS);
    println!("  Build time: {:.2}s  ({} total templates)",
        t0.elapsed().as_secs_f64(), templates.len());

    let mut clf = TemplateClassifier::new(Metric::NCC, (28, 28));
    for (label, img) in templates { clf.add_template(label, img); }
    let clf = Arc::new(clf);

    let pool = ThreadPoolBuilder::new().num_threads(num_cpus::get()).build().unwrap();
    let total = test_images.len();
    let start = Instant::now();
    let correct = pool.install(|| {
        test_images.par_iter().zip(test_labels.par_iter())
            .map(|(img, &label)| if clf.classify(img) == label { 1 } else { 0 })
            .sum::<usize>()
    });
    print_results(correct, total, start.elapsed(), pool.current_num_threads(),
        &format!("parallel / k-means ({}x10 templates)", TEMPLATES_PER_CLASS));
}

// ─── Output Helper ───────────────────────────────────────────────────────────

fn print_results(correct: usize, total: usize, dur: std::time::Duration, threads: usize, label: &str) {
    let accuracy   = correct as f32 / total as f32;
    let throughput = total as f64 / dur.as_secs_f64();
    println!("\n=== Results: {} ===", label);
    println!("  Threads       : {}", threads);
    println!("  Total samples : {}", total);
    println!("  Correct       : {}", correct);
    println!("  Accuracy      : {:.2}%", accuracy * 100.0);
    println!("  Time          : {:.3}s", dur.as_secs_f64());
    println!("  Throughput    : {:.2} images/sec", throughput);
}

// ─── Menu & Main ─────────────────────────────────────────────────────────────

fn print_menu() {
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║        MNIST Template Classifier Menu            ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║  1 — Sequential (mean templates)                 ║");
    println!("║  2 — Parallel: data parallelism (mean templates) ║");
    println!("║  3 — Parallel: data parallelism (k-means)        ║");
    println!("║  4 — Run all modes and compare                   ║");
    println!("║  0 — Exit                                        ║");
    println!("╚══════════════════════════════════════════════════╝");
    print!("Choice: ");
    io::stdout().flush().unwrap();
}

fn main() {
    println!("Loading MNIST...");
    let train_images = load_mnist_images("data/train-images.idx3-ubyte");
    let train_labels = load_mnist_labels("data/train-labels.idx1-ubyte");
    let test_images  = load_mnist_images("data/t10k-images.idx3-ubyte");
    let test_labels  = load_mnist_labels("data/t10k-labels.idx1-ubyte");
    println!("Loaded {} train / {} test.", train_images.len(), test_images.len());

    println!("Building mean templates...");
    let mean_templates = build_mean_templates(&train_images, &train_labels, 10);
    let mut clf_mean = TemplateClassifier::new(Metric::NCC, (28, 28));
    for (label, img) in mean_templates { clf_mean.add_template(label, img); }
    let clf_mean = Arc::new(clf_mean);

    loop {
        print_menu();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim() {
            "1" => run_sequential(&test_images, &test_labels, &clf_mean),
            "2" => run_parallel_mean(&test_images, &test_labels, Arc::clone(&clf_mean)),
            "3" => run_parallel_kmeans(&train_images, &train_labels, &test_images, &test_labels),
            "4" => {
                println!("\n>>> Running all three modes...");
                run_sequential(&test_images, &test_labels, &clf_mean);
                run_parallel_mean(&test_images, &test_labels, Arc::clone(&clf_mean));
                run_parallel_kmeans(&train_images, &train_labels, &test_images, &test_labels);
                println!("\n>>> All modes complete.");
            }
            "0" => { println!("Goodbye!"); break; }
            _   => println!("Invalid choice — enter 0 through 4."),
        }
    }
}
