import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const CONTENT_DIR = path.join(process.cwd(), 'content');

export interface ContentMeta {
  title: string;
  description: string;
  date: string;
  chapter?: number;
}
export interface ContentSummary extends ContentMeta { slug: string }
export interface NoteSummary extends ContentSummary { chapterSlug: string; chapter: number }
export interface Chapter { id: number; slug: string; title: string }
export interface Book { slug: string; title: string; author: string; description: string; chapters: Chapter[] }
export interface Project { slug: string; title: string; description: string; tech: string[]; github: string; demo: string; content: string }

function entries(directory: string, directories = false) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true })
    .filter(entry => directories ? entry.isDirectory() : entry.isFile() && entry.name.endsWith('.md'))
    .map(entry => entry.name).sort();
}

function text(value: unknown, field: string, filename: string, required = false): string {
  if (value === undefined && !required) return '';
  if (typeof value !== 'string' || (required && !value.trim())) {
    throw new Error(`${filename}: ${field} must be a ${required ? 'nonempty ' : ''}string`);
  }
  return value;
}

function readContent(filename: string) {
  const { data, content } = matter(fs.readFileSync(filename, 'utf8'));
  const date = text(data.date, 'date', filename, true);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(Date.parse(date)) ||
      new Date(date).toISOString().slice(0, 10) !== date) {
    throw new Error(`${filename}: date must be a valid YYYY-MM-DD date`);
  }
  if (data.chapter !== undefined && (!Number.isInteger(data.chapter) || data.chapter < 1)) {
    throw new Error(`${filename}: chapter must be a positive integer`);
  }
  const meta: ContentMeta = {
    title: text(data.title, 'title', filename, true),
    description: text(data.description, 'description', filename),
    date,
    chapter: data.chapter,
  };
  return { meta, content };
}

export function getPosts(): ContentSummary[] {
  const directory = path.join(CONTENT_DIR, 'blog');
  return entries(directory).map(filename => ({
    slug: filename.slice(0, -3), ...readContent(path.join(directory, filename)).meta,
  })).sort((a, b) => b.date.localeCompare(a.date) || a.slug.localeCompare(b.slug));
}

export function getPostBySlug(slug: string) {
  if (!getPosts().some(post => post.slug === slug)) return null;
  return readContent(path.join(CONTENT_DIR, 'blog', `${slug}.md`));
}

export function getBooks(): Book[] {
  const directory = path.join(CONTENT_DIR, 'notes');
  return entries(directory, true).flatMap(slug => {
    const filename = path.join(directory, slug, '_book.json');
    if (!fs.existsSync(filename)) return [];
    const data = JSON.parse(fs.readFileSync(filename, 'utf8'));
    if (!Array.isArray(data.chapters)) throw new Error(`${filename}: chapters must be an array`);
    const chapters: Chapter[] = data.chapters.map((chapter: Record<string, unknown>) => {
      if (!Number.isInteger(chapter.id) || Number(chapter.id) < 1) throw new Error(`${filename}: invalid chapter id`);
      return {
        id: Number(chapter.id),
        slug: text(chapter.slug, 'chapter slug', filename, true),
        title: text(chapter.title, 'chapter title', filename, true),
      };
    });
    if (new Set(chapters.map(chapter => chapter.id)).size !== chapters.length ||
        new Set(chapters.map(chapter => chapter.slug)).size !== chapters.length) {
      throw new Error(`${filename}: chapter ids and slugs must be unique`);
    }
    return [{
      slug, title: text(data.title, 'title', filename, true),
      author: text(data.author, 'author', filename, true),
      description: text(data.description, 'description', filename),
      chapters: chapters.sort((a, b) => a.id - b.id),
    }];
  });
}

export function getBookBySlug(slug: string) {
  return getBooks().find(book => book.slug === slug);
}

export function getBookNotes(bookSlug: string): NoteSummary[] {
  const directory = path.join(CONTENT_DIR, 'notes', bookSlug);
  return entries(directory, true).flatMap(chapterSlug =>
    entries(path.join(directory, chapterSlug)).map(filename => {
      const { meta } = readContent(path.join(directory, chapterSlug, filename));
      return {
        ...meta, chapter: meta.chapter ?? Number(chapterSlug.replace('chapter-', '')),
        chapterSlug, slug: filename.slice(0, -3),
      };
    })
  ).sort((a, b) => a.chapter - b.chapter || a.slug.localeCompare(b.slug));
}

export function getNotesForChapter(bookSlug: string, chapterSlug: string) {
  const book = getBookBySlug(bookSlug);
  const chapter = book?.chapters.find(item => item.slug === chapterSlug);
  return chapter ? getBookNotes(bookSlug).filter(note => note.chapter === chapter.id) : [];
}

export function notePath(bookSlug: string, note: Pick<NoteSummary, 'chapterSlug' | 'slug'>) {
  return `/notes/${bookSlug}/${note.chapterSlug}/${note.slug}`;
}

export function getNoteContent(bookSlug: string, chapterSlug: string, noteSlug: string) {
  const note = getBookNotes(bookSlug).find(item => item.chapterSlug === chapterSlug && item.slug === noteSlug);
  if (!note) return null;
  return readContent(path.join(CONTENT_DIR, 'notes', bookSlug, chapterSlug, `${noteSlug}.md`));
}

export function getProjects(): Project[] {
  const directory = path.join(CONTENT_DIR, 'projects');
  return entries(directory, true).flatMap(slug => {
    const filename = path.join(directory, slug, 'index.md');
    if (!fs.existsSync(filename)) return [];
    const { data, content } = matter(fs.readFileSync(filename, 'utf8'));
    if (!Array.isArray(data.tech) || !data.tech.every((item: unknown) => typeof item === 'string')) {
      throw new Error(`${filename}: tech must be an array of strings`);
    }
    return [{
      slug, title: text(data.title, 'title', filename, true),
      description: text(data.description, 'description', filename),
      tech: data.tech as string[],
      github: text(data.github, 'github', filename),
      demo: text(data.demo, 'demo', filename), content,
    }];
  });
}
