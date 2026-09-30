//! Words: what a question is searched with.

/// Words too common to search with.
pub const STOP: &[&str] = &[
    "what", "when", "where", "which", "while", "that", "this", "there", "their", "then", "than", "them", "they", "these", "those", "with", "from", "have", "were", "been", "will", "would", "could", "should", "about", "into", "just", "also", "more", "most", "some", "such", "very", "only", "your", "you", "does", "did", "how", "why", "and", "but", "the", "for", "not", "tell", "mean", "means", "say", "said", "says", "again", "else", "next", "please", "really", "thing", "things", "happen", "happens", "happened", "after", "before", "because", "now", "yes", "yep", "okay", "sure", "any", "all", "can", "may", "who", "its", "out", "get", "got", "let", "see", "too", "one", "two", "me", "go", "was", "are", "his", "her", "him", "she", "has", "had", "into", "our", "who", "whom", "whose", "a", "an", "of", "to", "in", "on", "at", "by", "is", "it", "be", "as", "or", "if", "so", "do", "no", "up", "we", "he", "i", "my",
];

/// A light stem: a plural's final "s" off (so "potions" finds "potion"), nothing cleverer - over-stemming merges words
/// that mean different things.
pub fn stem(word: &str) -> String {
    let w = word.to_lowercase();
    match w.strip_suffix('s') {
        Some(rest) if rest.chars().count() >= 3 && !rest.ends_with('s') => rest.to_string(),
        _ => w,
    }
}

/// A text's searchable words: lower-cased, stemmed, stop words and single letters out.
pub fn terms(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 2)
        .map(|w| w.to_lowercase())
        .filter(|w| !STOP.contains(&w.as_str()))
        .map(|w| stem(&w))
        .collect()
}

/// A question that counts, adds, compares or orders - where a model answering straight off goes wrong (a tie
/// reported, the ten points that broke it forgotten), so it is reasoned through first.
pub fn needs_thinking(message: &str) -> bool {
    let m = message.to_lowercase();
    ["how many", "how much", "in total", "total", "altogether", "score", "points", "tied", "a tie", "count", "difference", "older", "younger", "more than", "fewer", "less than", "how old", "how long", "in what order", "which came", "chronolog", "compare", "versus", "before or after", "who won", "winner"].iter().any(|k| m.contains(k))
}

/// The first line of a model's rewording, stripped of numbering and quotes; `fallback` if it is empty or runs on.
pub fn clean_rephrasing(raw: &str, fallback: &str) -> String {
    let line = raw.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let line = line.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == ' ');
    let line = line.trim_matches(|c: char| c == '"' || c == '\'' || c == '“' || c == '”' || c == '*').trim();
    if line.is_empty() || line.chars().count() > 300 {
        fallback.to_string()
    } else {
        line.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_keep_the_words_that_carry_meaning() {
        assert_eq!(terms("How many points did the houses have at the feast?"), vec!["many", "point", "house", "feast"]);
        assert_eq!(stem("glass"), "glass", "a double s is not a plural");
    }

    #[test]
    fn counting_and_comparing_are_reasoned_through() {
        assert!(needs_thinking("How many points did Gryffindor have?"));
        assert!(!needs_thinking("Who is Hagrid?"));
    }
}
