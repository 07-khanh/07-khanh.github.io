import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const contentDirectory = path.join(process.cwd(), 'content');

export interface BlogPost {
  slug: string;
  title: string;
  date: string;
  description: string;
  tags: string[];
  content: string;
}

export interface Project {
  slug: string;
  title: string;
  description: string;
  date: string;
  technologies: string[];
  githubUrl?: string;
  demoUrl?: string;
  featured?: boolean;
  content: string;
}

export interface ChapterMeta {
  slug: string;
  title: string;
  status: 'published' | 'draft' | 'planned';
}

export interface BookMeta {
  slug: string;
  title: string;
  author: string;
  description: string;
  chapters: ChapterMeta[];
}

export interface StudyNote {
  bookSlug: string;
  chapterSlug: string;
  title: string;
  content: string;
}

// BLOG PARSER
export function getAllBlogPosts(): BlogPost[] {
  const dir = path.join(contentDirectory, 'blog');
  if (!fs.existsSync(dir)) return [];
  const files = fs.readdirSync(dir);

  const posts = files
    .filter((f) => f.endsWith('.md') || f.endsWith('.mdx'))
    .map((file) => {
      const slug = file.replace(/\.mdx?$/, '');
      const raw = fs.readFileSync(path.join(dir, file), 'utf-8');
      const { data, content } = matter(raw);
      return {
        slug,
        title: data.title || slug,
        date: data.date || '',
        description: data.description || '',
        tags: data.tags || [],
        content,
      };
    });

  return posts.sort((a, b) => (a.date < b.date ? 1 : -1));
}

// PROJECTS PARSER
export function getAllProjects(): Project[] {
  const dir = path.join(contentDirectory, 'projects');
  if (!fs.existsSync(dir)) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  return entries
    .filter((e) => e.isDirectory())
    .map((entry) => {
      const slug = entry.name;
      const file = path.join(dir, slug, 'index.mdx');
      const raw = fs.readFileSync(file, 'utf-8');
      const { data, content } = matter(raw);
      return {
        slug,
        title: data.title || slug,
        description: data.description || '',
        date: data.date || '',
        technologies: data.technologies || [],
        githubUrl: data.githubUrl,
        demoUrl: data.demoUrl,
        featured: Boolean(data.featured),
        content,
      };
    })
    .sort((a, b) => (a.date < b.date ? 1 : -1));
}

// STUDY NOTES PARSER
export function getAllBooks(): BookMeta[] {
  const dir = path.join(contentDirectory, 'notes');
  if (!fs.existsSync(dir)) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  return entries
    .filter((e) => e.isDirectory())
    .map((entry) => {
      const bookSlug = entry.name;
      const metaFile = path.join(dir, bookSlug, 'meta.json');
      if (!fs.existsSync(metaFile)) return null;
      const rawMeta = fs.readFileSync(metaFile, 'utf-8');
      const meta = JSON.parse(rawMeta);
      return {
        slug: bookSlug,
        ...meta,
      };
    })
    .filter(Boolean) as BookMeta[];
}

export function getNoteContent(bookSlug: string, chapterSlug: string): StudyNote | null {
  const filePath = path.join(contentDirectory, 'notes', bookSlug, `${chapterSlug}.mdx`);
  if (!fs.existsSync(filePath)) return null;

  const raw = fs.readFileSync(filePath, 'utf-8');
  const { data, content } = matter(raw);

  return {
    bookSlug,
    chapterSlug,
    title: data.title || chapterSlug,
    content,
  };
}
