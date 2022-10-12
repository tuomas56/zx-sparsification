use std::collections::HashSet;
use petgraph::prelude as px;
use rand::seq::IteratorRandom;

#[derive(Debug, Clone)]
struct GeometricSeries {
    current: f32,
    scale: f32,
    steps: usize
}

impl GeometricSeries {
    fn new(start: f32, stop: f32, steps: usize) -> Self {
        GeometricSeries {
            current: start.ln(), steps,
            scale: (stop.ln() - start.ln()) / steps as f32
        }
    }
}

impl Iterator for GeometricSeries {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.steps == 0 {
            None
        } else {
            let out = self.current.exp();
            self.current += self.scale;
            self.steps -= 1;
            Some(out)
        }
    }
}

struct ComplementFinder<R: rand::Rng, T: Iterator<Item = f32>> {
    graph: px::UnGraph<(), ()>,
    current: HashSet<px::NodeIndex>,
    fitness: usize,
    rng: R,
    temperature: T
}

impl<R: rand::Rng, T: Iterator<Item = f32>> ComplementFinder<R, T> {
    fn new<N, E>(graph: &px::UnGraph<N, E>, rng: R, temperature: T) -> Self {
        ComplementFinder {
            graph: graph.map(|_, _| (), |_, _| ()),
            current: HashSet::new(),
            fitness: graph.edge_count(),
            rng, temperature
        }
    }

    fn toggle_node(&mut self, node: px::NodeIndex) {
        let mut present = false;
        for &other in &self.current {
            if other == node {
                present = true;
                continue
            }

            if let Some(edge) = self.graph.find_edge(node, other) {
                self.graph.remove_edge(edge);
            } else {
                self.graph.add_edge(node, other, ());
            }
        }

        if !present {
            self.current.insert(node);
        } else {
            self.current.remove(&node);
        }
    }

    fn step(&mut self, temp: f32) {
        let node = self.graph
            .node_indices()
            .choose(&mut self.rng)
            .unwrap();

        self.toggle_node(node);
        let new_fitness = self.graph.edge_count();

        if new_fitness < self.fitness {
            self.fitness = new_fitness;
            return
        }

        let prob = ((new_fitness - self.fitness) as f32 / temp).exp().recip();
        if self.rng.gen::<f32>() < prob {
            self.fitness = new_fitness;
        } else {
            self.toggle_node(node);
        }
    }

    fn run(&mut self, quiet: bool) {
        let mut step = 0;
        let original = self.fitness;
        while let Some(temp) = self.temperature.next() {
            if !quiet && step % 1000 == 0 {
                println!(
                    "step = {:?}, temp = {:.2?}, fitness = {:?}, ratio = {:.2?}", 
                    step, temp, self.fitness, self.fitness as f32 / original as f32
                );
            }
            self.step(temp);
            step += 1;
        }

        if !quiet {
            println!(
                "final: fitness = {:?}, ratio = {:.2?}", 
                self.fitness, self.fitness as f32 / original as f32
            );
        }
    }
}

struct PivotFinder<R: rand::Rng, T: Iterator<Item = f32>> {
    graph: px::UnGraph<(), ()>,
    left: HashSet<px::NodeIndex>,
    right: HashSet<px::NodeIndex>,
    fitness: usize,
    rng: R,
    temperature: T
}

impl<R: rand::Rng, T: Iterator<Item = f32>> PivotFinder<R, T> {
    fn new<N, E>(graph: &px::UnGraph<N, E>, rng: R, temperature: T) -> Self {
        PivotFinder {
            graph: graph.map(|_, _| (), |_, _| ()),
            left: HashSet::new(),
            right: HashSet::new(),
            fitness: graph.edge_count(),
            rng, temperature
        }
    }

    fn toggle_node(&mut self, node: px::NodeIndex, left: bool) {
        let (this_side, other_side) = if left {
            (&mut self.left, &self.right)
        } else {
            (&mut self.right, &self.left)
        };

        for &other in other_side {
            if let Some(edge) = self.graph.find_edge(node, other) {
                self.graph.remove_edge(edge);
            } else {
                self.graph.add_edge(node, other, ());
            }
        }

        if !this_side.contains(&node) {
            this_side.insert(node);
        } else {
            this_side.remove(&node);
        }
    }

    fn step(&mut self, temp: f32) {
        let left = self.rng.gen_bool(0.5);

        let node = loop {
            let prop = self.graph
                .node_indices()
                .choose(&mut self.rng)
                .unwrap();

            if left && !self.right.contains(&prop) {
                break prop
            } else if !left && !self.left.contains(&prop) {
                break prop
            }
        };
        

        self.toggle_node(node, left);
        let new_fitness = self.graph.edge_count();

        if new_fitness < self.fitness {
            self.fitness = new_fitness;
            return
        }

        let prob = ((new_fitness - self.fitness) as f32 / temp).exp().recip();
        if self.rng.gen::<f32>() < prob {
            self.fitness = new_fitness;
        } else {
            self.toggle_node(node, left);
        }
    }

    fn run(&mut self, quiet: bool) {
        let mut step = 0;
        let original = self.fitness;
        while let Some(temp) = self.temperature.next() {
            if !quiet && step % 1000 == 0 {
                println!(
                    "step = {:?}, temp = {:.2?}, fitness = {:?}, ratio = {:.2?}", 
                    step, temp, self.fitness, self.fitness as f32 / original as f32
                );
            }
            self.step(temp);
            step += 1;
        }

        if !quiet {
            println!(
                "final: fitness = {:?}, ratio = {:.2?}", 
                self.fitness, self.fitness as f32 / original as f32
            );
        }
    }
}

#[derive(Debug)]
enum SparsifierMove {
    Complement(HashSet<px::NodeIndex>),
    Pivot(HashSet<px::NodeIndex>, HashSet<px::NodeIndex>)
}

struct Sparsifier<R: rand::Rng, T: Iterator<Item = f32> + Clone> {
    graph: px::UnGraph<(), ()>,
    moves: Vec<SparsifierMove>,
    rng: R,
    temperature: T,
    alpha: f32,
    beta: f32
}

impl<R: rand::Rng, T: Iterator<Item = f32> + Clone> Sparsifier<R, T > {
    fn new<N, E>(graph: &px::UnGraph<N, E>, temperature: T, rng: R, alpha: f32, beta: f32) -> Self {
        Sparsifier {
            graph: graph.map(|_, _| (), |_, _| ()),
            moves: Vec::new(),
            temperature, rng, alpha, beta
        }
    }

    fn cost(&self, graph: &px::UnGraph<(), ()>) -> f32 {
        graph.node_count() as f32 * self.alpha + graph.edge_count() as f32 * self.beta
    }

    fn run(&mut self, quiet: bool) {
        let original = self.cost(&self.graph);
        let originale = self.graph.edge_count();
        loop {
            if !quiet {
                println!("=> cost: {:.2?}, edges: {:?}", self.cost(&self.graph), self.graph.edge_count());
            }

            let mut cfinder = ComplementFinder::new(
                &self.graph, &mut self.rng, self.temperature.clone()
            );
            cfinder.run(quiet);
            let cgraph = cfinder.graph;
            let cloc = cfinder.current;
            let complement_cost = 1.0 + self.cost(&cgraph);

            let mut pfinder = PivotFinder::new(
                &self.graph, &mut self.rng, self.temperature.clone()
            );
            pfinder.run(quiet);
            let pgraph = pfinder.graph;
            let ploc = (pfinder.left, pfinder.right);
            let pivot_cost = 2.0 + self.cost(&pgraph);

            if complement_cost.min(pivot_cost) >= self.cost(&self.graph) {
                if !quiet {
                    println!("=> no improvement found, stopping");
                }

                break
            }

            if complement_cost < pivot_cost {
                if !quiet {
                    println!("=> doing complement on {} vertices", cloc.len());
                }

                self.graph = cgraph;
                self.moves.push(SparsifierMove::Complement(cloc));
            } else {
                if !quiet {
                    println!("=> doing pivot between {} and {} vertices", ploc.0.len(), ploc.1.len());
                }

                self.graph = pgraph;
                self.moves.push(SparsifierMove::Pivot(ploc.0, ploc.1));
            }
        }

        if !quiet {
            println!(
                "=> cost ratio: {:?}, edge ratio: {:?}, move list:", 
                self.cost(&self.graph) / original, 
                self.graph.edge_count() as f32 / originale as f32
            );
            for m in &self.moves {
                match m {
                    SparsifierMove::Complement(c) => println!(
                        "  - complement on {:?} vertices", c.len()
                    ),
                    SparsifierMove::Pivot(a, b) => println!(
                        "  - pivot between {} and {} vertices", a.len(), b.len()
                    )
                }
            }
        }
    }
}

fn gnp(n: usize, p: f32) -> px::UnGraph<(), ()> {
    let mut graph = px::UnGraph::new_undirected();
    
    let nodes = (0..n)
        .map(|_| graph.add_node(()))
        .collect::<Vec<_>>();

    for &a in &nodes {
        for &b in &nodes {
            if rand::random::<f32>() < p {
                graph.add_edge(a, b, ());
            }
        }
    }

    graph
}

fn main() {
    let graph = gnp(100, 0.5);

    let mut sparsifier = Sparsifier::new(
        &graph, 
        GeometricSeries::new(3000.0, 1.0, 2000),
        rand::thread_rng(),
        0.25, 0.02
    );

    let before = std::time::Instant::now();
    sparsifier.run(false);
    println!("{:?} elapsed", before.elapsed());
}
