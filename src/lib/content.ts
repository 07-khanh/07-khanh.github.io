import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const CONTENT_DIR = path.join(process.cwd(), 'content');

export function getPosts() {
  const blogDir = path.join(CONTENT_DIR, 'blog');
  if (!fs.existsSync(blogDir)) return [];
  const files = fs.readdirSync(blogDir).filter(f => f.endsWith('.md'));
  
  return files.map(filename => {
    const filePath = path.join(blogDir, filename);
    const { data } = matter(fs.readFileSync(filePath, 'utf8'));
    return {
      slug: filename.replace('.md', ''),
      title: data.title || 'Untitled',
      date: data.date || '',
      description: data.description || '',
    };
  }).sort((a, b) => (a.date > b.date ? -1 : 1));
}

export function getPostBySlug(slug: string) {
  const filePath = path.join(CONTENT_DIR, 'blog', `${slug}.md`);
  if (!fs.existsSync(filePath)) return null;
  const { data, content } = matter(fs.readFileSync(filePath, 'utf8'));
  return { meta: data, content };
}

export function getBooks() {
  const notesDir = path.join(CONTENT_DIR, 'notes');
  if (!fs.existsSync(notesDir)) return [];
  const folders = fs.readdirSync(notesDir).filter(f => fs.statSync(path.join(notesDir, f)).isDirectory());
  
  return folders.map(folder => {
    const metaPath = path.join(notesDir, folder, '_book.json');
    if (!fs.existsSync(metaPath)) return null;
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    return { slug: folder, ...meta };
  }).filter(Boolean);
}

export function getBookBySlug(slug: string) {
  const books = getBooks();
  return books.find(b => b?.slug === slug);
}

export function getNotesForChapter(bookSlug: string, chapterSlug: string) {
  const chapterDir = path.join(CONTENT_DIR, 'notes', bookSlug, chapterSlug);
  if (!fs.existsSync(chapterDir)) return [];
  const files = fs.readdirSync(chapterDir).filter(f => f.endsWith('.md'));
  
  return files.map(filename => {
    const filePath = path.join(chapterDir, filename);
    const { data } = matter(fs.readFileSync(filePath, 'utf8'));
    return {
      slug: filename.replace('.md', ''),
      title: data.title || 'Untitled',
      date: data.date || '',
      description: data.description || '',
    };
  });
}

export function getNoteContent(bookSlug: string, chapterSlug: string, noteSlug: string) {
  const filePath = path.join(CONTENT_DIR, 'notes', bookSlug, chapterSlug, `${noteSlug}.md`);
  if (!fs.existsSync(filePath)) return null;
  const { data, content } = matter(fs.readFileSync(filePath, 'utf8'));
  return { meta: data, content };
}

export function getProjects() {
  const projectsDir = path.join(CONTENT_DIR, 'projects');
  if (!fs.existsSync(projectsDir)) return [];
  const folders = fs.readdirSync(projectsDir).filter(f => fs.statSync(path.join(projectsDir, f)).isDirectory());
  
  return folders.map(folder => {
    const filePath = path.join(projectsDir, folder, 'index.md');
    if (!fs.existsSync(filePath)) return null;
    const { data, content } = matter(fs.readFileSync(filePath, 'utf8'));
    return {
      slug: folder,
      title: data.title || 'Untitled',
      description: data.description || '',
      tech: data.tech || [],
      github: data.github || '',
      demo: data.demo || '',
      content
    };
  }).filter(Boolean);
}
