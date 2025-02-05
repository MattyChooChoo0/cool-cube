import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import random
import ctypes
import pyrr
from math import *

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640

ASPECT_RATIO = 640/480

GLOBAL_X = np.array([1,0,0], dtype=np.float32)
GLOBAL_Y = np.array([0,1,0], dtype=np.float32)
GLOBAL_Z = np.array([0,0,1], dtype=np.float32)

FOVX = 60

FOVY = FOVX/ASPECT_RATIO


print('Y FOV: ' + str(FOVY))

def getCameraDirectionVector(yaw, pitch):
    x = 1
    y = 0
    z = 0
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)

    x = cos(pitch)*cos(yaw)
    y = sin(pitch)*cos(yaw)
    z = sin(yaw)

    return (x, y, -1*z)

class Cube():
    def __init__(self, position=[0,0,0], eulers=[0,0,0]):
        self.position = position
        self.eulers = eulers
    def getModelMatrix(self):
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_eulers(
                eulers = np.radians(self.eulers),
                dtype=np.float32
            )
        )
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(
                vec = self.position,
                dtype=np.float32
            )
        )
        return model_transform
class Player:
    def __init__(self, position=[-0.1,0,-0.1], eulers=[0,0,0]):
        self.position = position
        self.eulers = eulers
        self.mesh = Triangle()
        self.yaw = -90
        self.pitch = 0
        self.turning = {'left':False, 'right':False}
        self.movement = {'forward':False, 'backward':False, 'left':False, 'right':False}
        self.forwardvelocity = 0
        self.material = Material('gfx/wood.jpeg')
        self.sidevelocity = 0
        theta = self.eulers[2]
        phi = self.eulers[1]

        self.forwards = np.array(
            [
                np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(phi))
            ],
            dtype = np.float32
        )
    def getModelMatrix(self):
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_eulers(
                eulers = np.radians(self.eulers),
                dtype=np.float32
            )
        )
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(
                vec = self.position,
                dtype=np.float32
            )
        )
        return model_transform

    def draw(self, shader):
        self.material.use()
        model_transform = self.getModelMatrix()
        lookat_matrix = self.getLookAtMatrix()
        modelMatrixReference = glGetUniformLocation(shader,'modelMatrix')
        lookatMatrixReference = glGetUniformLocation(shader,'lookatMatrix')
        glUniformMatrix4fv(modelMatrixReference, 1, GL_FALSE, self.getModelMatrix())
        glUniformMatrix4fv(lookatMatrixReference, 1, GL_FALSE, lookat_matrix)
        glBindVertexArray(self.mesh.vao)
        #glDrawArrays(GL_QUADS, 0, self.mesh.vertex_count)
    def update(self):
        if self.turning['left']:
            self.eulers[2] += 1
        if self.turning['right']:
            self.eulers[2] -= 1
        theta = self.eulers[2]
        phi = self.eulers[1]

        self.forwards = np.array(
            [
                np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(phi)),
                np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
            ],
            dtype = np.float32
        )
        forwardsplaceholder = (self.forwards[2],self.forwards[0],self.forwards[1])
        self.right = np.cross(forwardsplaceholder, GLOBAL_Z)
        self.right = (-1*self.right[1], -1*self.right[2], -1*self.right[0])

        self.up = np.cross(self.right, self.forwards)
        if self.pitch >= 360:
            self.pitch -= 360
        if self.pitch < 0:
            self.pitch += 360

        if self.yaw >= 360:
            self.yaw -= 360
        if self.yaw < 0:
            self.yaw += 360
        #print(self.yaw)
        if self.movement['left']:
            self.position[2] += self.right[2]*-0.02
            self.position[0] += self.right[0]*-0.02
            print(self.right)

        if self.movement['right']:
            self.position[2] += self.right[2]*0.02
            self.position[0] += self.right[0]*0.02

        if self.movement['forward']:
            self.position[2] -= self.forwards[2]*0.02
            self.position[1] += self.forwards[1]*0.02
            self.position[0] -= self.forwards[0]*0.02
        if self.movement['backward']:
            self.position[2] -= self.forwards[2]*-0.02
            self.position[1] += self.forwards[1]*-0.02
            self.position[0] -= self.forwards[0]*-0.02

        if not (self.movement['forward'] or self.movement['backward']):
            if self.forwardvelocity != 0:
                self.forwardvelocity /= 1.1
                if abs(self.forwardvelocity) <= 0.0001:
                    self.forwardvelocity = 0

        if not (self.movement['left'] or self.movement['right']):
            if self.sidevelocity != 0:
                self.sidevelocity /= 1.1
                if abs(self.sidevelocity) <= 0.0001:
                    self.sidevelocity = 0




    def getLookAtMatrix(self):
        pos = self.position
        target = [0,0,0]

        target = pos-self.forwards

        up = [0,1,0]
        lookmatrix = pyrr.matrix44.create_look_at(pos, target, up, dtype=None)


        return lookmatrix


class App:
    def __init__(self):
        self.displayFlags = pygame.DOUBLEBUF | pygame.OPENGL
        self.win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), self.displayFlags)
        pygame.display.set_caption("OpenGL project")
        self.pitch = 0
        self.yaw = 0
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                    pygame.GL_CONTEXT_PROFILE_CORE)

        self.setup_opengl()
        self.shader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")
        glUseProgram(self.shader)
        self.triangle = Triangle()
        self.running = False
        self.material = Material('gfx/dirt.png')
        self.cube = Cube(position=[0,0,-4])
        self.player = Player()

        glEnable(GL_DEPTH_TEST);
        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = FOVY, aspect = 640/480,
            near = 0.1, far = 10, dtype=np.float32
        )
        self.clock = pygame.time.Clock()

        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projectionMatrix"),
            1, GL_FALSE, projection_transform
        )
        self.modelMatrixReference = glGetUniformLocation(self.shader,'modelMatrix')
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"),0)
    def setup_opengl(self):
        glClearColor(0.1, 0.0, 0.1, 1.0)
    def createShader(self, vertex_filepath, fragment_filepath):
        with open(vertex_filepath,'r') as f:
            vertex_src = f.readlines()

        with open(fragment_filepath,'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))
        return shader
    def run(self):
        self.running = True
        while self.running:
            #print(self.player.position)
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.player.turning['left'] = True
                    if event.key == pygame.K_RIGHT:
                        self.player.turning['right'] = True
                    if event.key == pygame.K_w:
                        self.player.movement['forward'] = True
                    if event.key == pygame.K_s:
                        self.player.movement['backward'] = True
                    if event.key == pygame.K_a:
                        self.player.movement['left'] = True
                    if event.key == pygame.K_d:
                        self.player.movement['right'] = True

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        self.player.turning['left'] = False
                    if event.key == pygame.K_RIGHT:
                        self.player.turning['right'] = False
                    if event.key == pygame.K_w:
                        self.player.movement['forward'] = False
                    if event.key == pygame.K_s:
                        self.player.movement['backward'] = False
                    if event.key == pygame.K_a:
                        self.player.movement['left'] = False
                    if event.key == pygame.K_d:
                        self.player.movement['right'] = False
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            #self.player.position[0] = 0
            self.material.use()
            #print(getCameraDirectionVector(self.pitch, 0))
            glUseProgram(self.shader)
            glUniformMatrix4fv(self.modelMatrixReference, 1, GL_FALSE, self.cube.getModelMatrix())
            glBindVertexArray(self.triangle.vao)

            glDrawArrays(GL_QUADS, 0, self.triangle.vertex_count)
            self.player.update()
            self.player.draw(self.shader)
            self.player.getLookAtMatrix()


            pygame.display.flip()
            pygame.display.set_caption(str(self.clock.get_fps()))
    def quit(self):
        """ cleanup the app, run exit code """
        self.running = False

class Triangle():
    def __init__(self):
        vertices = (
            #front and back
            -0.5, -0.5, 0.5, 0.0, 1.0,
             0.5, -0.5, 0.5, 1.0, 1.0,
             0.5,  0.5, 0.5, 1.0, 0.0,
            -0.5,  0.5, 0.5, 0.0, 0.0,

            -0.5, -0.5, -0.5, 0.0, 1.0,
             0.5, -0.5, -0.5, 1.0, 1.0,
             0.5,  0.5, -0.5, 1.0, 0.0,
            -0.5,  0.5, -0.5, 0.0, 0.0,
            #top and bottom
            -0.5,  0.5, -0.5, 0.0, 0.0,
             0.5,  0.5, -0.5, 1.0, 0.0,
             0.5,  0.5,  0.5, 1.0, 1.0,
            -0.5,  0.5,  0.5, 0.0, 1.0,

            -0.5, -0.5, -0.5, 0.0, 1.0,
             0.5, -0.5, -0.5, 0.0, 0.0,
             0.5, -0.5,  0.5, 1.0, 0.0,
            -0.5, -0.5,  0.5, 1.0, 1.0,
            #left and right
            -0.5,  0.5, -0.5, 0.0, 0.0,
            -0.5,  0.5,  0.5, 1.0, 0.0,
            -0.5, -0.5,  0.5, 1.0, 1.0,
            -0.5, -0.5, -0.5, 0.0, 1.0,

             0.5,  0.5, -0.5, 0.0, 0.0,
             0.5,  0.5,  0.5, 1.0, 0.0,
             0.5, -0.5,  0.5, 1.0, 1.0,
             0.5, -0.5, -0.5, 0.0, 1.0

        )

        vertices = np.array(vertices, dtype=np.float32)

        self.vertex_count = 24

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))

    def setUniform(self, variablename, value, shader):
        uniformreference = glGetUniformLocation(shader, variablename)
        glUniform1f(uniformreference, value)

class Material:
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pygame.image.load(filepath).convert()
        image_width, image_height = image.get_rect().size
        image_data = pygame.image.tostring(image, "RGBA")
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        glGenerateMipmap(GL_TEXTURE_2D)
    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
    def destroy(self):
        glDeleteTextures(1, (self.texture,))
my_app = App()
my_app.run()
pygame.quit()

