use crate::idx::Idx;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub distances: Vec<f32>,
    pub labels: Vec<Idx>,
}

impl SearchResult {
    pub fn new(distances: Vec<f32>, labels: Vec<Idx>) -> Self {
        Self { distances, labels }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn n_queries(&self) -> usize {
        if self.labels.is_empty() {
            0
        } else {
            self.distances.len() / self.labels.len() * self.labels.len()
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Idx, f32)> + '_ {
        self.labels
            .iter()
            .copied()
            .zip(self.distances.iter().copied())
    }

    pub fn labels(&self) -> impl Iterator<Item = Option<u64>> + '_ {
        self.labels.iter().map(|&l| l.get())
    }

    pub fn get(&self, query_idx: usize, k: usize) -> Option<(Vec<Idx>, Vec<f32>)> {
        let start = query_idx * k;
        let end = start + k;
        if end <= self.labels.len() && end <= self.distances.len() {
            Some((
                self.labels[start..end].to_vec(),
                self.distances[start..end].to_vec(),
            ))
        } else {
            None
        }
    }

    pub fn first_label(&self) -> Option<u64> {
        self.labels.first().and_then(|&l| l.get())
    }

    pub fn first_distance(&self) -> Option<f32> {
        self.distances.first().copied()
    }
}

#[derive(Debug, Clone)]
pub struct RangeSearchResult {
    pub labels: Vec<Idx>,
    pub distances: Vec<f32>,
    pub lims: Vec<usize>,
}

impl RangeSearchResult {
    pub fn new(labels: Vec<Idx>, distances: Vec<f32>, lims: Vec<usize>) -> Self {
        Self {
            labels,
            distances,
            lims,
        }
    }

    pub fn n_queries(&self) -> usize {
        self.lims.len().saturating_sub(1)
    }

    pub fn total_results(&self) -> usize {
        self.labels.len()
    }

    pub fn get(&self, query_idx: usize) -> Option<(Vec<Idx>, Vec<f32>)> {
        let start = *self.lims.get(query_idx)?;
        let end = *self.lims.get(query_idx + 1)?;
        Some((
            self.labels[start..end].to_vec(),
            self.distances[start..end].to_vec(),
        ))
    }

    pub fn iter(&self) -> RangeSearchResultIter<'_> {
        RangeSearchResultIter {
            result: self,
            query_idx: 0,
        }
    }
}

pub struct RangeSearchResultIter<'a> {
    result: &'a RangeSearchResult,
    query_idx: usize,
}

impl<'a> Iterator for RangeSearchResultIter<'a> {
    type Item = (Vec<Idx>, Vec<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.result.get(self.query_idx)?;
        self.query_idx += 1;
        Some(result)
    }
}

#[derive(Debug, Clone)]
pub struct BinarySearchResult {
    pub distances: Vec<i32>,
    pub labels: Vec<Idx>,
}

impl BinarySearchResult {
    pub fn new(distances: Vec<i32>, labels: Vec<Idx>) -> Self {
        Self { distances, labels }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Idx, i32)> + '_ {
        self.labels
            .iter()
            .copied()
            .zip(self.distances.iter().copied())
    }

    pub fn get(&self, query_idx: usize, k: usize) -> Option<(Vec<Idx>, Vec<i32>)> {
        let start = query_idx * k;
        let end = start + k;
        if end <= self.labels.len() && end <= self.distances.len() {
            Some((
                self.labels[start..end].to_vec(),
                self.distances[start..end].to_vec(),
            ))
        } else {
            None
        }
    }
}
